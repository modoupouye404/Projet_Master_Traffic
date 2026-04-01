from flask import Flask, jsonify, request
import requests
import json
import csv

app = Flask(__name__)

# URL de l'API RandomUser
API_URL = "https://randomuser.me/api/"

@app.route('/api/get_all_users', methods=['GET'])
def get_all_users():
    try:
        # Récupérer les paramètres facultatifs depuis la requête
        num_users = request.args.get('results', default=10, type=int)  # Par défaut, 10 utilisateurs
        nationality = request.args.get('nat')  # Nationalité (facultatif)

        # Construire les paramètres pour l'API RandomUser
        params = {
            "results": num_users,
            "nat": nationality
        }

        # Faire une requête GET à l'API RandomUser
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Vérifie les erreurs HTTP

        # Récupérer toutes les données de la réponse
        data = response.json()
        users = data['results']  # Liste des utilisateurs

        # Sauvegarder toutes les données dans un fichier JSON
        json_file_path = "randomuser_all_data.json"
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(users, json_file, ensure_ascii=False, indent=4)

        # Sauvegarder toutes les données dans un fichier CSV
        csv_file_path = "randomuser_all_data.csv"
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)

            # Écrire l'en-tête du fichier CSV
            # Dynamiquement en fonction des clés du premier utilisateur
            if users:
                headers = list(users[0].keys())
                writer.writerow(headers)

                # Écrire les données de chaque utilisateur
                for user in users:
                    row = [json.dumps(user.get(key, "")) for key in headers]  # Convertir les objets en chaîne JSON
                    writer.writerow(row)

        # Retourner un message au client avec les fichiers sauvegardés
        return jsonify({
            "message": f"{num_users} utilisateur(s) récupéré(s) avec succès.",
            "files_saved": {
                "json": json_file_path,
                "csv": csv_file_path
            },
            "example_users": users[:3]  # Exemple : 3 premiers utilisateurs
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Erreur lors de la requête : {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)