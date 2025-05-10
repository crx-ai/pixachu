import subprocess


def main():
    print("Running post-install script...")
    print("Installing pre-commit hooks...")
    subprocess.run(["pre-commit", "install"], check=True)
    print("Done!")


if __name__ == "__main__":
    main()
