import sys
from app.cli import main
if __name__ == "__main__":
    sys.argv = ["app.cli", "all"]
    main()
