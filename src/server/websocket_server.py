"""
Enhanced WebSocket Server for China Multi-Lingual ASR System.

This module extends the existing WhisperLive WebSocket server with
integrated ASR capabilities and intelligent language routing.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any
import websockets
from websockets.server import WebSocketServerProtocol

# TODO: Import actual implementations when available
# from src.core.integrated_asr import IntegratedASR, ASRConfig
# from src.core.output_adapter import OutputAdapter


class WebSocketServer:
    """
    Enhanced WebSocket server with integrated multi-lingual ASR capabilities.
    
    This server maintains compatibility with existing WhisperLive clients
    while providing intelligent language routing and optimized performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the WebSocket server."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Server configuration
        self.host = config.get("server", {}).get("websocket", {}).get("host", "0.0.0.0")
        self.port = config.get("server", {}).get("websocket", {}).get("port", 9090)
        self.max_connections = config.get("server", {}).get("websocket", {}).get("max_connections", 100)
        
        # Initialize integrated ASR system
        self.integrated_asr = None
        self.output_adapter = None
        
        # Connection management
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Server statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_transcriptions": 0,
            "total_audio_duration": 0.0,
            "uptime_start": time.time()
        }
        
        self.is_running = False
    
    async def initialize(self) -> bool:
        """Initialize the server and all components."""
        try:
            # TODO: Initialize IntegratedASR
            # asr_config = ASRConfig(
            #     whisper_model_path=self.config["models"]["whisper"]["model_path"],
            #     kimi_model_path=self.config["models"]["kimi"]["model_path"],
            #     device=self.config["models"]["whisper"]["device"],
            #     confidence_threshold=self.config["language_routing"]["confidence_threshold"],
            #     kimi_languages=self.config["language_routing"]["kimi_languages"]
            # )
            # self.integrated_asr = IntegratedASR(asr_config)
            # if not await self.integrated_asr.initialize():
            #     return False
            
            # TODO: Initialize OutputAdapter
            # self.output_adapter = OutputAdapter()
            
            self.logger.info("WebSocket server initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket server: {e}")
            return False
    
    async def start(self):
        """Start the WebSocket server."""
        if not await self.initialize():
            raise RuntimeError("Failed to initialize server")
        
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=1024*1024*10,  # 10MB max message size
            ping_interval=30,
            ping_timeout=10
        ):
            self.is_running = True
            self.logger.info("WebSocket server started successfully")
            
            # Keep server running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                self.logger.info("Server shutdown requested")
            finally:
                await self.cleanup()
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{time.time()}"
        
        try:
            # Check connection limits
            if len(self.active_connections) >= self.max_connections:
                await websocket.send(json.dumps({
                    "status": "ERROR",
                    "message": "Server at maximum capacity",
                    "code": 1002
                }))
                await websocket.close()
                return
            
            # Register connection
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                "connected_at": time.time(),
                "remote_address": websocket.remote_address,
                "transcription_count": 0,
                "total_audio_duration": 0.0
            }
            
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            
            self.logger.info(f"New connection: {client_id}")
            
            # Handle client session
            await self.handle_client_session(websocket, client_id)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed: {client_id}")
        except Exception as e:
            self.logger.error(f"Error handling connection {client_id}: {e}")
        finally:
            # Cleanup connection
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                del self.connection_metadata[client_id]
                self.stats["active_connections"] -= 1
    
    async def handle_client_session(self, websocket: WebSocketServerProtocol, client_id: str):
        """Handle a complete client session."""
        # Wait for initial configuration
        config_message = await websocket.recv()
        
        try:
            client_config = json.loads(config_message)
            self.logger.debug(f"Client config: {client_config}")
            
            # Validate and process client configuration
            session_config = self.process_client_config(client_config)
            
            # Send server ready message
            await websocket.send(json.dumps({
                "message": "SERVER_READY",
                "status": "SERVER_READY",
                "backend": "integrated_asr",
                "uid": client_config.get("uid", client_id),
                "code": 0
            }))
            
            # Handle audio streaming
            await self.handle_audio_stream(websocket, client_id, session_config)
            
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "status": "ERROR",
                "message": "Invalid configuration format",
                "code": 1009
            }))
        except Exception as e:
            self.logger.error(f"Session error for {client_id}: {e}")
            await websocket.send(json.dumps({
                "status": "ERROR", 
                "message": "Internal server error",
                "code": 1001
            }))
    
    def process_client_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate client configuration."""
        return {
            "uid": config.get("uid"),
            "language": config.get("language"),
            "task": config.get("task", "transcribe"),
            "model": config.get("model", "large-v3"),
            "use_vad": config.get("use_vad", True),
            "initial_prompt": config.get("initial_prompt"),
            "vad_parameters": config.get("vad_parameters", {"threshold": 0.5})
        }
    
    async def handle_audio_stream(self, websocket: WebSocketServerProtocol, 
                                client_id: str, session_config: Dict[str, Any]):
        """Handle streaming audio transcription."""
        audio_buffer = bytearray()
        session_start = time.time()
        
        try:
            while True:
                # Receive audio data
                message = await websocket.recv()
                
                if message == b"END_OF_AUDIO" or message == "END_OF_AUDIO":
                    self.logger.info(f"End of audio received from {client_id}")
                    
                    # Process final buffer if any
                    if audio_buffer:
                        await self.process_audio_chunk(
                            websocket, client_id, bytes(audio_buffer), 
                            session_config, is_final=True
                        )
                    
                    break
                
                # Handle binary audio data
                if isinstance(message, bytes):
                    audio_buffer.extend(message)
                    
                    # Process chunks when buffer reaches threshold
                    if len(audio_buffer) >= 32000:  # ~2 seconds at 16kHz
                        await self.process_audio_chunk(
                            websocket, client_id, bytes(audio_buffer), session_config
                        )
                        audio_buffer.clear()
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed during audio stream: {client_id}")
        except Exception as e:
            self.logger.error(f"Audio stream error for {client_id}: {e}")
        finally:
            # Update session statistics
            session_duration = time.time() - session_start
            self.connection_metadata[client_id]["total_audio_duration"] += session_duration
            self.stats["total_audio_duration"] += session_duration
    
    async def process_audio_chunk(self, websocket: WebSocketServerProtocol, 
                                client_id: str, audio_data: bytes, 
                                session_config: Dict[str, Any], is_final: bool = False):
        """Process an audio chunk through the integrated ASR system."""
        try:
            import numpy as np
            
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # TODO: Use IntegratedASR for transcription
            # result = await self.integrated_asr.transcribe(
            #     audio_array,
            #     language=session_config.get("language"),
            #     initial_prompt=session_config.get("initial_prompt"),
            #     vad=session_config.get("use_vad", True),
            #     vad_params=session_config.get("vad_parameters")
            # )
            
            # TODO: Convert to WhisperLive format
            # whisperlive_output = self.output_adapter.to_whisperlive_format(result)
            
            # Placeholder response
            placeholder_segments = [{
                "start": "0.000",
                "end": f"{len(audio_array)/16000:.3f}",
                "text": "[PLACEHOLDER] Transcription result from integrated ASR"
            }]
            
            # Send transcription result
            response = {
                "uid": session_config.get("uid", client_id),
                "code": 0,
                "status": "RESULT",
                "segments": placeholder_segments,
                "is_end": is_final,
                "metadata": {
                    "engine": "integrated_asr",
                    "processing_time": 0.1,
                    "audio_duration": len(audio_array) / 16000
                }
            }
            
            await websocket.send(json.dumps(response, ensure_ascii=False))
            
            # Update statistics
            self.connection_metadata[client_id]["transcription_count"] += 1
            self.stats["total_transcriptions"] += 1
            
        except Exception as e:
            self.logger.error(f"Audio processing error for {client_id}: {e}")
            
            # Send error response
            error_response = {
                "uid": session_config.get("uid", client_id),
                "code": 1001,
                "status": "ERROR",
                "message": "Audio processing failed"
            }
            await websocket.send(json.dumps(error_response))
    
    async def get_server_stats(self) -> Dict[str, Any]:
        """Get current server statistics."""
        uptime = time.time() - self.stats["uptime_start"]
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "average_session_duration": (
                self.stats["total_audio_duration"] / self.stats["total_connections"]
                if self.stats["total_connections"] > 0 else 0
            ),
            "transcriptions_per_hour": (
                self.stats["total_transcriptions"] / (uptime / 3600)
                if uptime > 0 else 0
            )
        }
    
    async def cleanup(self):
        """Cleanup server resources."""
        self.logger.info("Cleaning up WebSocket server...")
        
        # Close all active connections
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send(json.dumps({
                    "status": "DISCONNECT",
                    "message": "Server shutting down",
                    "code": 1000
                }))
                await websocket.close()
            except Exception as e:
                self.logger.warning(f"Error closing connection {client_id}: {e}")
        
        # Cleanup integrated ASR
        if self.integrated_asr:
            # TODO: Cleanup integrated ASR
            # self.integrated_asr.cleanup()
            pass
        
        self.is_running = False
        self.logger.info("WebSocket server cleanup completed")


# CLI entry point
async def main():
    """Main entry point for running the WebSocket server."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description="China Multi-Lingual ASR WebSocket Server")
    parser.add_argument("--config", default="config/development.yaml", help="Configuration file path")
    parser.add_argument("--host", help="Server host (overrides config)")
    parser.add_argument("--port", type=int, help="Server port (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override with command line arguments
    if args.host:
        config.setdefault("server", {}).setdefault("websocket", {})["host"] = args.host
    if args.port:
        config.setdefault("server", {}).setdefault("websocket", {})["port"] = args.port
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format=config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Create and start server
    server = WebSocketServer(config)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main()) 