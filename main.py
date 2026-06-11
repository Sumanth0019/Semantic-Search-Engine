import sys
import os

def print_usage():
    print("Usage:")
    print("  python main.py ingest  — build vector index")
    print("  python main.py search  — interactive search")
    print("  python main.py api     — start FastAPI server")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "ingest":
        import ingest
        ingest.run()

    elif mode == "search":
        import search
        reranker = __import__("reranker")
        reranker.get_reranker()
        print("Models loaded. Ready.\n")
        while True:
            q = input("Search: ").strip()
            if q.lower() in ("quit","exit","q"):
                break
            if not q:
                continue
            results = search.full_search(q)
            search.format_results(results, q)

    elif mode == "api":
        import uvicorn
        print("Starting API server...")
        print("Swagger UI: http://localhost:8000/docs")
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )

    else:
        print(f"Unknown mode: '{mode}'")
        print_usage()

if __name__ == "__main__":
    main()
