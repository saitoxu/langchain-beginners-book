# LangChain Beginners Book

Python開発環境（Docker Compose + uv）

## 必要なツール

- Docker Desktop
- uv（ホスト環境用）

## セットアップ手順

### 1. uvのインストール（ホスト環境）

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. ホスト環境のセットアップ

エディタでのsyntax highlightingと補完を有効にするため、ホスト環境にもPython環境を構築します。

```bash
# Python 3.12の仮想環境を作成
uv venv --python 3.12

# 仮想環境を有効化
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 依存関係をインストール
uv pip install -r requirements.txt
```

### 3. Docker環境のビルドと起動

```bash
# Dockerイメージをビルド
docker compose build

# コンテナに入る
docker compose run --rm app bash
```

## 使い方

### Docker環境での作業

```bash
# コンテナに入る
docker compose run --rm app bash

# Python REPLを起動
python

# スクリプトを実行
python your_script.py
```

### ホスト環境での作業

```bash
# 仮想環境を有効化
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate     # Windows

# Pythonスクリプトを実行
python your_script.py
```

## パッケージの追加

新しいパッケージを追加する場合：

1. `requirements.txt`に追加
2. ホスト環境で更新：
   ```bash
   uv pip install -r requirements.txt
   ```
3. Dockerイメージを再ビルド：
   ```bash
   docker compose build
   ```

## エディタの設定

### VS Code

`.vscode/settings.json`を作成：

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. Settings → Project → Python Interpreterを開く
2. 歯車アイコン → Add
3. Existing environmentを選択
4. `.venv/bin/python`を選択

## トラブルシューティング

### uvコマンドが見つからない

```bash
# PATHに追加されているか確認
echo $PATH

# 手動でPATHに追加（bash/zsh）
export PATH="$HOME/.local/bin:$PATH"
```

### Dockerコンテナ内でパッケージが見つからない

```bash
# イメージを再ビルド
docker compose build --no-cache
```
