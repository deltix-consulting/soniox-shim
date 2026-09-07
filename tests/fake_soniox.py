"""A stand-in for wss://stt-rt.soniox.com: records what the shim sends, replies with a script."""

import asyncio
import json
import threading

from websockets.asyncio.server import serve


class FakeSoniox:
    def __init__(self, script: list[dict], hang: bool = False):
        self.script, self.hang = script, hang
        self.config: dict | None = None
        self.audio = b""
        self.url = ""
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None

    async def _handler(self, ws):
        self.config = json.loads(await ws.recv())
        async for frame in ws:
            if frame == "" or frame == b"":
                break
            self.audio += frame
        if self.hang:
            await asyncio.sleep(60)
        for msg in self.script:
            await ws.send(json.dumps(msg))

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            self.url = f"ws://127.0.0.1:{port}"
            self._ready.set()
            await self._stop.wait()

    def __enter__(self):
        threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True).start()
        self._ready.wait(5)
        return self

    def __exit__(self, *_):
        self._loop.call_soon_threadsafe(self._stop.set)
