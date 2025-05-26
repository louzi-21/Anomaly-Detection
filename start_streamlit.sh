#!/bin/bash

echo "⏳ Attente que MySQL soit prêt..."
until mysqladmin ping -h"mysql" --silent; do
    sleep 2
done

echo "✅ MySQL est prêt, démarrage de Streamlit"
exec streamlit run dashboard.py --server.port=8501 --server.headless=true
