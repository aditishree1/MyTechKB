def main() -> None:
    import uvicorn

    from mytechkb.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "mytechkb.api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
