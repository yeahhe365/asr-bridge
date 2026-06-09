from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "fun_asr_openai_proxy.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
