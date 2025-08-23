# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
from typing import Any

from verl.tools.utils.mcp_clients.utils import TokenBucket, mcp2openai

logger = logging.getLogger(__name__)


class MCPClientManager:
    rootServerName = "mcpServers"
    initialized = False
    clients = []
    tool_client_mapping = {}
    rate_limiter = None

    async def initialize(self, config_path, rate_limit: float = 10.0):
        if self.initialized:
            return
        """Initialize the MCP Client Manager and start all clients"""
        
        try:
            # 使用标准的 mcp 库而不是 fastmcp
            from mcp import Client, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.sse import sse_client
            from mcp.client.streamable_http import streamablehttp_client
            from mcp import ClientSession
        except ImportError as e:
            logger.error(f"Failed to import mcp library: {e}")
            logger.info("Please install the standard mcp library: pip install mcp")
            raise
            
        result = self._load_config(config_path)
        logger.info(f"Loaded MCP config: {result}")
        
        servers = result[self.rootServerName]
        
        for server_name, server_config in servers.items():
            try:
                if "url" in server_config:
                    # HTTP/SSE based server (like microsandbox)
                    from mcp.client.streamable_http import streamablehttp_client
                    
                    # 创建连接参数
                    url = server_config["url"]
                    headers = server_config.get("headers", {})
                    timeout = server_config.get("timeout", 30)
                    sse_read_timeout = server_config.get("sse_read_timeout", 300)
                    
                    logger.info(f"Connecting to HTTP MCP server {server_name} at {url}")
                    
                    # 使用标准 mcp 库的 streamablehttp_client
                    self.clients.append({
                        'name': server_name,
                        'type': 'http',
                        'url': url,
                        'headers': headers,
                        'timeout': timeout,
                        'sse_read_timeout': sse_read_timeout
                    })
                    
                elif "command" in server_config:
                    # Stdio based server
                    command = server_config["command"]
                    args = server_config.get("args", [])
                    env = server_config.get("env", {})
                    
                    logger.info(f"Connecting to stdio MCP server {server_name}: {command} {args}")
                    
                    self.clients.append({
                        'name': server_name,
                        'type': 'stdio',
                        'command': command,
                        'args': args,
                        'env': env
                    })
                    
            except Exception as e:
                logger.error(f"Failed to setup client for {server_name}: {e}")
                continue

        # Initialize rate limiter
        self.rate_limiter = TokenBucket(rate_limit)
        self.initialized = True
        logger.info(f"Successfully initialized MCPClientManager with {len(self.clients)} clients")

    async def call_tool(self, tool_name, parameters, timeout):
        """Call a tool with proper timeout handling"""
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        client_config = self.get_client_with_tool_name(tool_name)
        if not client_config:
            raise ValueError(f"No client found for tool: {tool_name}")
            
        try:
            # 使用 asyncio.wait_for 来实现超时控制
            result = await asyncio.wait_for(
                self._call_tool_with_client(client_config, tool_name, parameters),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool call {tool_name} timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Tool call {tool_name} failed: {e}")
            raise

    async def _call_tool_with_client(self, client_config, tool_name, parameters):
        """Execute tool call with specific client"""
        if client_config['type'] == 'http':
            return await self._call_http_tool(client_config, tool_name, parameters)
        elif client_config['type'] == 'stdio':
            return await self._call_stdio_tool(client_config, tool_name, parameters)
        else:
            raise ValueError(f"Unsupported client type: {client_config['type']}")

    async def _call_http_tool(self, client_config, tool_name, parameters):
        """Call tool via HTTP/SSE transport"""
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession
        
        async with streamablehttp_client(
            url=client_config['url'],
            headers=client_config.get('headers', {}),
            timeout=client_config.get('timeout', 30),
            sse_read_timeout=client_config.get('sse_read_timeout', 300)
        ) as (read, write, get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, parameters)
                return result

    async def _call_stdio_tool(self, client_config, tool_name, parameters):
        """Call tool via stdio transport"""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters
        
        server_params = StdioServerParameters(
            command=client_config['command'],
            args=client_config['args'],
            env=client_config.get('env')
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, parameters)
                return result

    async def fetch_tool_schemas(self, tool_selected_list: list[str]) -> list[dict]:
        """Fetch tool schemas from all clients"""
        tool_schemas = []
        
        for client_config in self.clients:
            try:
                tools = await self._fetch_tools_from_client(client_config)
                for tool in tools:
                    tool_name = tool.name
                    if not tool_selected_list or tool_name in tool_selected_list:
                        self.tool_client_mapping[tool_name] = client_config
                        tool_schemas.append(mcp2openai(tool))
                        logger.debug(f"Registered tool {tool_name} from client {client_config['name']}")
                        
            except Exception as e:
                logger.error(f"Failed to fetch tools from client {client_config.get('name', 'unknown')}: {e}")
                continue

        logger.info(f"Fetched {len(tool_schemas)} tool schemas")
        return tool_schemas

    async def _fetch_tools_from_client(self, client_config):
        """Fetch tools from a specific client"""
        if client_config['type'] == 'http':
            return await self._fetch_http_tools(client_config)
        elif client_config['type'] == 'stdio':
            return await self._fetch_stdio_tools(client_config)
        else:
            raise ValueError(f"Unsupported client type: {client_config['type']}")

    async def _fetch_http_tools(self, client_config):
        """Fetch tools via HTTP/SSE transport"""
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession
        
        async with streamablehttp_client(
            url=client_config['url'],
            headers=client_config.get('headers', {}),
            timeout=client_config.get('timeout', 30),
            sse_read_timeout=client_config.get('sse_read_timeout', 300)
        ) as (read, write, get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return tools_result.tools

    async def _fetch_stdio_tools(self, client_config):
        """Fetch tools via stdio transport"""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters
        
        server_params = StdioServerParameters(
            command=client_config['command'],
            args=client_config['args'],
            env=client_config.get('env')
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return tools_result.tools

    def get_client_with_tool_name(self, tool_name: str):
        """Get client configuration for a specific tool"""
        return self.tool_client_mapping.get(tool_name)

    def _load_config(self, file: str) -> dict[str, Any]:
        try:
            with open(file) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f'the "{file}" file was not found')
        except Exception:
            logger.error(f'there was an error reading the "{file}" file')

        return {}


# 创建修复版本的实例
ClientManager = MCPClientManager()
