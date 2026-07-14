from flask import Flask, request, jsonify
import datetime
import os

app = Flask(__name__)

DATASET_ID = os.environ.get("COPERNICUS_DATASET_ID", "cmems_mod_med_wav_anfc_4.2km_PT1H-i")
VARIABLES = ["VHM0", "VTPK", "VMDR"]


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/wave")
def wave():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "INVALID_PARAMS", "message": "lat e lon richiesti come numeri"}), 400

    # Parametro opzionale "date" (YYYY-MM-DD) per interrogare un giorno futuro
    # entro l'orizzonte di forecast del dataset anfc (di norma alcuni giorni
    # avanti). Senza il parametro il comportamento resta l'istante attuale,
    # invariato rispetto a prima.
    target = datetime.datetime.utcnow()
    date_param = request.args.get("date")
    if date_param:
        try:
            target_date = datetime.datetime.strptime(date_param, "%Y-%m-%d").date()
            target = datetime.datetime.combine(target_date, datetime.time(12, 0))
        except ValueError:
            return jsonify({"error": "INVALID_PARAMS", "message": "date deve essere in formato YYYY-MM-DD"}), 400

    try:
        import copernicusmarine
    except ImportError:
        return jsonify({"error": "MISSING_DEPENDENCY", "message": "copernicusmarine non installato"}), 500

    start = target - datetime.timedelta(hours=3)
    end = target + datetime.timedelta(hours=3)

    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            variables=VARIABLES,
            minimum_longitude=lon,
            maximum_longitude=lon,
            minimum_latitude=lat,
            maximum_latitude=lat,
            start_datetime=start.isoformat(),
            end_datetime=end.isoformat(),
            coordinates_selection_method="nearest",
        )
        point = ds.sel(time=target, method="nearest")
        point = point.squeeze()

        result = {
            "source": "copernicus-marine-cmems",
            "datasetId": DATASET_ID,
            "waveHeightM": round(float(point["VHM0"].values), 2),
            "wavePeriodS": round(float(point["VTPK"].values), 1),
            "waveDirectionDeg": round(float(point["VMDR"].values), 0),
            "timestamp": str(point["time"].values),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "CMEMS_FETCH_ERROR", "message": str(e)}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
