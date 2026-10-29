"""Regression for Windows closing an API listener after a queued client reset."""
import asyncio
import os
import socket
import struct
import unittest

import uvicorn
from aiwf import engine_api


@unittest.skipUnless(os.name == "nt", "Windows socket-reset behavior")
class WindowsApiLoopTest(unittest.TestCase):
    def test_listener_survives_queued_connection_reset(self):
        config = uvicorn.Config(lambda scope, receive, send: None, loop=engine_api._server_loop())
        factory = config.get_loop_factory()

        async def probe():
            async def handle(reader, writer):
                writer.close()
                try:
                    await writer.wait_closed()
                except (OSError, ConnectionError):
                    pass

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            first = second = None
            try:
                # Block the event loop while one connection fills the pending
                # accept and a second waits in the queue and sends a reset.
                first = socket.create_connection(("127.0.0.1", port), timeout=2)
                second = socket.create_connection(("127.0.0.1", port), timeout=2)
                second.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("HH", 1, 0))
                second.close()
                second = None
                await asyncio.sleep(0.3)
                _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 2)
                writer.close()
                try:
                    await writer.wait_closed()
                except (OSError, ConnectionError):
                    pass
            finally:
                if first is not None:
                    first.close()
                if second is not None:
                    second.close()
                server.close()
                await server.wait_closed()

        asyncio.run(probe(), loop_factory=factory)


if __name__ == "__main__":
    unittest.main()
