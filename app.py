from flask import Flask, request, jsonify
import datetime
import os

app = Flask(__name__)

DATASET_ID = os.environ.get("COPERNICUS_DATASET_ID", "cmems_mod_med_wav_anfc_4.2km_PT1H-i")
VARIABLES = ["VHM0", "VTPK", "VMDR"]

# Molti spot sono a ridosso della costa: sulla griglia a 4.2km del modello
# la cella esatta spesso cade su "terra" (valori NaN). Se succede, allarghiamo
# progressivamente il raggio di ricerca e prendiamo la cella di mare valida
# più vicina al punto richiesto, invece di restituire NaN.
SEARCH_RADII_DEG = [0.05, 0.1, 0.2, 0.4]


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
        result = fetch_nearest_valid_wave(lat, lon, target, start, end)
        if result is None:
            return jsonify({
                "error": "NO_SEA_DATA_NEARBY",
                "message": f"Nessuna cella di mare valida trovata entro {SEARCH_RADII_DEG[-1]}° da lat={lat}, lon={lon}.",
            }), 502
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "CMEMS_FETCH_ERROR", "message": str(e)}), 502


def fetch_nearest_valid_wave(lat, lon, target, start, end):
    """
    Interroga il dataset su un riquadro attorno al punto richiesto (invece
    del solo punto esatto) e restituisce la cella di mare valida (non-NaN)
    più vicina, allargando il raggio se serve. Ritorna None se non trova
    nulla di valido entro il raggio massimo.
    """
    import copernicusmarine
    import numpy as np

    for radius in SEARCH_RADII_DEG:
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            variables=VARIABLES,
            minimum_longitude=lon - radius,
            maximum_longitude=lon + radius,
            minimum_latitude=lat - radius,
            maximum_latitude=lat + radius,
            start_datetime=start.isoformat(),
            end_datetime=end.isoformat(),
        )
        try:
            snapshot = ds.sel(time=target, method="nearest").squeeze()

            height = snapshot["VHM0"].values
            if height.ndim == 0:
                # Riquadro troppo piccolo per contenere più di una cella: la
                # cella unica è terra (NaN) o mare (valida), nessuna scelta da fare.
                if not np.isnan(height):
                    return _point_result(snapshot)
                continue

            valid = ~np.isnan(height)
            if not valid.any():
                continue

            lats = snapshot["latitude"].values
            lons = snapshot["longitude"].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            dist_sq = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
            dist_sq = np.where(valid, dist_sq, np.inf)
            flat_idx = np.argmin(dist_sq)
            iy, ix = np.unravel_index(flat_idx, dist_sq.shape)
            nearest = snapshot.isel(latitude=iy, longitude=ix).load()
            return _point_result(nearest)
        finally:
            # Libera subito la memoria del dataset: su hosting free-tier a
            # risorse limitate, tenerne aperti più d'uno per richiesta
            # (un tentativo per ogni raggio) esaurisce facilmente la RAM.
            ds.close()

    return None


def _point_result(point):
    return {
        "source": "copernicus-marine-cmems",
        "datasetId": DATASET_ID,
        "waveHeightM": round(float(point["VHM0"].values), 2),
        "wavePeriodS": round(float(point["VTPK"].values), 1),
        "waveDirectionDeg": round(float(point["VMDR"].values), 0),
        "timestamp": str(point["time"].values),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
