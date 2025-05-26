# Détection d'Anomalies - Application Flask + Streamlit

Cette application est conçue pour détecter des anomalies dans une base de données produits, avec :
- 🔐 Authentification sécurisée (admin / admin123)
- 📊 Dashboard interactif via Streamlit intégré dans Flask
- ⚙️ Déploiement facile avec Docker Compose

---

## 🔧 Lancer le projet

### Prérequis :
- Docker Desktop installé
- Port 5000 (Flask) et 8501 (Streamlit) disponibles

### Étapes :
1. Placez-vous dans le dossier du projet :
```bash
cd anomaly_app
```

2. Lancez l'application :
```bash
docker-compose up --build #premiere fois
```
docker-compose up 
---

## 🌐 Accès

- Interface utilisateur : http://localhost:5000
- Dashboard Streamlit (via iframe) : http://localhost:5000/dashboard
- Accès direct Streamlit (optionnel) : http://localhost:8501

---
## Base de données

- MySQL accessible dans Docker à `host = "mysql"`, port 3306
- Utilisateur : root
- Mot de passe : rootpass
- Base : pfe

## 👤 Identifiants de connexion
- **Utilisateur** : `admin`
- **Mot de passe** : `admin123`

---

## 📁 Structure du projet

```
anomaly_app/
├── app.py                      # Point d'entrée Flask
├── Dockerfile                  # Image Docker unifiée
├── docker-compose.yml          # Configuration des services Flask + 
├── start_streamlit.sh
├── requirements.txt            # Dépendances Python
├── auth/routes.py              # Authentification
├── views/routes.py             # Route /dashboard
├── templates/                  # HTML (base, login, dashboard)
└── dashboard.py                # Ton dashboard Streamlit (à renommer si différent)
```

---

## 📬 Support

