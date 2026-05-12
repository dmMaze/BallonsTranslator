const repo = "https://github.com/CoSciBlog/BallonsTranslator-vibe.git"
const branch = "dev"

module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          `git remote set-url origin ${repo}`,
          `git fetch origin ${branch}`,
          `git pull --ff-only origin ${branch}`
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          HF_HUB_ENABLE_HF_TRANSFER: "1"
        },
        message: [
          "python -m pip install --upgrade pip wheel setuptools",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "env"
      }
    }
  ]
}
