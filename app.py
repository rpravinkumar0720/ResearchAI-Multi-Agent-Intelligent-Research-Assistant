"""Entry point for ResearchAI – Multi-Agent Intelligent Research Assistant."""

from frontend.interface import build_interface


def main() -> None:
    app = build_interface()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
