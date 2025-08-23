# Copyright 2025 Bytedance Ltd. and/or its affiliates
# 
# 简化的 HTTP 客户端，直接调用 microsandbox API
# 绕过复杂的 MCP 协议栈

import asyncio
import json
import logging
import aiohttp
from typing import Any

from verl.tools.utils.mcp_clients.utils import TokenBucket

logger = logging.getLogger(__name__)


class SimpleHttpClientManager:
    """简化的 HTTP 客户端管理器，直接调用 microsandbox API"""
    
    def __init__(self):
        self.initialized = False
        self.base_url = None
        self.rate_limiter = None
        self.session = None
        
    async def initialize(self, config_path, rate_limit: float = 10.0):
        if self.initialized:
            return
            
        # 直接使用 microsandbox 的 HTTP API
        self.base_url = "http://localhost:5555"
        self.rate_limiter = TokenBucket(rate_limit)
        
        # 创建 aiohttp session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )
        
        self.initialized = True
        logger.info("SimpleHttpClientManager initialized")
        
    async def call_tool(self, tool_name, parameters, timeout):
        """直接调用 microsandbox API"""
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)
            
        if tool_name == "sandbox_start":
            return await self._sandbox_start(parameters, timeout)
        elif tool_name == "sandbox_run_code":
            return await self._sandbox_run_code(parameters, timeout)
        elif tool_name == "sandbox_stop":
            return await self._sandbox_stop(parameters, timeout)
        else:
            # 对于其他工具，尝试通用的 MCP 调用
            return await self._generic_mcp_call(tool_name, parameters, timeout)
    
    async def _sandbox_start(self, parameters, timeout):
        """启动 sandbox"""
        try:
            # 确保包含必要的配置
            arguments = parameters.copy()
            if 'config' not in arguments:
                arguments['config'] = {
                    'image': 'microsandbox/python',
                    'cpus': 1,
                    'memory': 512,
                    'envs': ['PYTHONPATH=/workspace'],
                    'ports': [],
                    'volumes': []
                }
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "sandbox_start",
                    "arguments": arguments
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # 构造与 MCP 兼容的响应
                    return self._create_mock_result(result)
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error(f"sandbox_start failed: {e}")
            raise
    
    async def _sandbox_run_code(self, parameters, timeout):
        """在 sandbox 中运行代码"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "sandbox_run_code",
                    "arguments": parameters
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._create_mock_result(result)
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error(f"sandbox_run_code failed: {e}")
            raise
            
    async def _sandbox_stop(self, parameters, timeout):
        """停止 sandbox"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "sandbox_stop",
                    "arguments": parameters
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._create_mock_result(result)
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error(f"sandbox_stop failed: {e}")
            raise
    
    async def _generic_mcp_call(self, tool_name, parameters, timeout):
        """通用的 MCP 工具调用"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._create_mock_result(result)
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                    
        except Exception as e:
            logger.error(f"Generic MCP call {tool_name} failed: {e}")
            raise
    
    def _create_mock_result(self, json_result):
        """创建与 MCP CallToolResult 兼容的响应"""
        class MockResult:
            def __init__(self, content):
                if isinstance(content, dict) and "result" in content:
                    # 从 JSON-RPC 响应中提取内容
                    result_data = content["result"]
                    if isinstance(result_data, dict) and "content" in result_data:
                        self.content = result_data["content"]
                    else:
                        self.content = [MockTextContent(str(result_data))]
                else:
                    self.content = [MockTextContent(str(content))]
        
        class MockTextContent:
            def __init__(self, text):
                self.type = "text"
                self.text = text
        
        return MockResult(json_result)
    
    async def fetch_tool_schemas(self, tool_selected_list: list[str]) -> list[dict]:
        """返回固定的工具 schema"""
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "sandbox_start",
                    "description": "Start a new sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string"},
                            "sandbox": {"type": "string"}
                        },
                        "required": ["sandbox", "namespace"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sandbox_run_code",
                    "description": "Execute code in sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "language": {"type": "string"},
                            "namespace": {"type": "string"},
                            "sandbox": {"type": "string"}
                        },
                        "required": ["sandbox", "namespace", "code", "language"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sandbox_stop",
                    "description": "Stop a sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string"},
                            "sandbox": {"type": "string"}
                        },
                        "required": ["sandbox", "namespace"]
                    }
                }
            }
        ]
        
        if tool_selected_list:
            # 过滤只包含选定的工具
            filtered_schemas = []
            for schema in schemas:
                if schema["function"]["name"] in tool_selected_list:
                    filtered_schemas.append(schema)
            return filtered_schemas
        
        return schemas
    
    def get_client_with_tool_name(self, tool_name: str):
        """为了兼容性，返回 self"""
        return self
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()


# 创建简化版本的实例
ClientManager = SimpleHttpClientManager()
