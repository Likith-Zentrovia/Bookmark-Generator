"""Entry point for running the web app: python -m bookmark_generator.web"""

from .app import create_app

app = create_app()
app.run(host="127.0.0.1", port=5000, debug=True)
