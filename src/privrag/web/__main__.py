def main() -> None:
    import uvicorn

    uvicorn.run(
        "privrag.web.app:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
