"""A stand-in for https://api.soniox.com/v1: records what the shim sends, replies with a script."""

import asyncio
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request, Response, UploadFile


class FakeSoniox:
    def __init__(
        self,
        tokens: list[dict] | None = None,
        upload_status: int = 201,
        job_error: str | None = None,
        hang: bool = False,
    ):
        self.tokens = tokens or []
        self.upload_status, self.job_error, self.hang = upload_status, job_error, hang
        self.audio = b""
        self.filename = ""
        self.job: dict = {}
        self.polls = 0
        self.deleted: list[str] = []
        self.url = ""
        self._server: uvicorn.Server | None = None
        app = self.app = FastAPI()

        @app.post("/files")
        async def upload(file: UploadFile):
            self.audio, self.filename = await file.read(), file.filename or ""
            if self.upload_status >= 400:
                return Response('{"message":"balance exhausted"}', self.upload_status)
            return {"id": "f1"}

        @app.post("/transcriptions")
        async def create(req: Request):
            self.job = await req.json()
            return {"id": "t1", "status": "queued"}

        @app.get("/transcriptions/t1")
        async def status():
            self.polls += 1
            if self.hang:
                await asyncio.sleep(60)
            if self.job_error:
                return {"id": "t1", "status": "error", "error_message": self.job_error}
            return {"id": "t1", "status": "queued" if self.polls < 2 else "completed"}

        @app.get("/transcriptions/t1/transcript")
        async def transcript():
            return {"id": "t1", "text": "", "tokens": self.tokens}

        @app.delete("/transcriptions/{tid}")
        @app.delete("/files/{tid}")
        async def delete(tid: str):
            self.deleted.append(tid)
            return Response(status_code=204)

    def __enter__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(uvicorn.Config(self.app, port=port, log_level="warning"))
        threading.Thread(target=self._server.run, daemon=True).start()
        while not self._server.started:
            time.sleep(0.01)
        return self

    def __exit__(self, *_):
        self._server.should_exit = True
