#!/bin/bash
# Served With a Smile: local MongoDB for the episodes (the Atlas clusters in the env no longer resolve, 30 Oct).
# Starts mongod on data/mongo if it is not already answering, then pushes every unsynced episode.
cd "$(dirname "$0")"
MONGOD=$(command -v mongod || ls /opt/homebrew/bin/mongod 2>/dev/null)
if [ -z "$MONGOD" ]; then echo "mongod not installed: brew trust mongodb/brew && brew install mongodb-community"; exit 1; fi
mkdir -p data/mongo
if ! nc -z localhost 27017 2>/dev/null; then
  "$MONGOD" --dbpath data/mongo --bind_ip 127.0.0.1 --port 27017 --fork --logpath data/mongo/mongod.log >/dev/null && echo "mongod started on localhost:27017 (data/mongo)"
else
  echo "mongod already answering on 27017"
fi
sleep 1
.venv/bin/python episodes.py push
