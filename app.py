from flask import Flask, request, jsonify
from zoneinfo import ZoneInfo
import datetime
import os

app = Flask(__name__)

DATASET_ID = os.environ.get("COPERNICUS_DATASET_ID", "cmems_mod_med_wav_anfc_4.2km_PT1H-i")
VARIABLES = ["VHM0", "VTPK", "VMDR"]

# Dataset fisico CMEMS per il livello del mare (sea surface height, variabile
# "zos"), stessa famiglia/risoluzione (4.2km) e stesso account del dataset
# onde sopra. Verificato via ricerca (luglio 2026): esiste anche una
# variante "detided" separata, il che conferma che questo prodotto standard
# include il segnale di marea vero (non solo dinamica generica).
TIDE_DATASET_ID = os.environ.get("COPERNICUS_TIDE_DATASET_ID", "cmems_mod_med_phy-ssh_anfc_4.2km-2D_PT1H-m")
TIDE_VARIABLES = ["zos"]
ROME_TZ = ZoneInfo("Europe/Rome")

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


@app.route("/tide")
def tide():
    """
    Livello del mare orario (marea) per l'intera giornata richiesta,
    dataset fisico CMEMS (variabile "zos"). A differenza di /wave, che
    prende un solo istante, qui si scarica una volta la serie oraria
    completa del giorno: la cella di mare valida più vicina si sceglie su
    un istante rappresentativo (mezzogiorno locale) per evitare 24 query
    separate sulla stessa area.
    """
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "INVALID_PARAMS", "message": "lat e lon richiesti come numeri"}), 400

    date_param = request.args.get("date")
    if date_param:
        try:
            target_date = datetime.datetime.strptime(date_param, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "INVALID_PARAMS", "message": "date deve essere in formato YYYY-MM-DD"}), 400
    else:
        target_date = datetime.datetime.now(ROME_TZ).date()

    # Il "giorno" richiesto è un giorno di calendario Europe/Rome (quello che
    # percepisce l'utente), non UTC: costruiamo i confini in ora locale e li
    # convertiamo in UTC (naive) solo per interrogare il dataset, che lavora
    # in UTC come /wave.
    day_start_local = datetime.datetime.combine(target_date, datetime.time(0, 0), tzinfo=ROME_TZ)
    day_end_local = datetime.datetime.combine(target_date, datetime.time(23, 0), tzinfo=ROME_TZ)
    midday_local = datetime.datetime.combine(target_date, datetime.time(12, 0), tzinfo=ROME_TZ)

    day_start_utc = day_start_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    day_end_utc = day_end_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    midday_utc = midday_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    try:
        import copernicusmarine
    except ImportError:
        return jsonify({"error": "MISSING_DEPENDENCY", "message": "copernicusmarine non installato"}), 500

    try:
        result = fetch_nearest_valid_tide(lat, lon, midday_utc, day_start_utc, day_end_utc, target_date)
        if result is None:
            return jsonify({
                "error": "NO_SEA_DATA_NEARBY",
                "message": f"Nessuna cella di mare valida trovata entro {SEARCH_RADII_DEG[-1]}° da lat={lat}, lon={lon}.",
            }), 502
        result["date"] = target_date.isoformat()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "CMEMS_FETCH_ERROR", "message": str(e)}), 502


def fetch_nearest_valid_tide(lat, lon, midday_utc, day_start_utc, day_end_utc, target_date):
    """
    Come fetch_nearest_valid_wave, ma invece di un solo istante estrae
    l'intera serie oraria del giorno per la cella di mare valida più vicina.
    """
    import copernicusmarine
    import numpy as np

    for radius in SEARCH_RADII_DEG:
        ds = copernicusmarine.open_dataset(
            dataset_id=TIDE_DATASET_ID,
            variables=TIDE_VARIABLES,
            minimum_longitude=lon - radius,
            maximum_longitude=lon + radius,
            minimum_latitude=lat - radius,
            maximum_latitude=lat + radius,
            start_datetime=(day_start_utc - datetime.timedelta(hours=1)).isoformat(),
            end_datetime=(day_end_utc + datetime.timedelta(hours=1)).isoformat(),
        )
        try:
            snapshot = ds.sel(time=midday_utc, method="nearest").squeeze()
            level = snapshot["zos"].values

            if level.ndim == 0:
                if np.isnan(level):
                    continue
                series = ds["zos"].squeeze()
            else:
                valid = ~np.isnan(level)
                if not valid.any():
                    continue
                lats = snapshot["latitude"].values
                lons = snapshot["longitude"].values
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                dist_sq = (lat_grid - lat) ** 2 + (lon_grid - lon) ** 2
                dist_sq = np.where(valid, dist_sq, np.inf)
                flat_idx = np.argmin(dist_sq)
                iy, ix = np.unravel_index(flat_idx, dist_sq.shape)
                series = ds["zos"].isel(latitude=iy, longitude=ix)

            hourly = _hourly_series(series.load(), target_date)
            if not hourly:
                continue
            return {
                "source": "copernicus-marine-cmems",
                "datasetId": TIDE_DATASET_ID,
                "hourly": hourly,
            }
        finally:
            ds.close()

    return None


def _hourly_series(series, target_date):
    """Converte la serie oraria (tempi UTC del dataset) in coppie
    {time: "HH:MM", level: metri}, con l'orario espresso in Europe/Rome —
    così sia il matching lato Node sia la visualizzazione lato utente usano
    la stessa ora "di parete" percepita, senza dover fare conversioni altrove.
    Il margine di ±1h nella query (per sicurezza sui bordi) può includere
    punti del giorno prima/dopo: qui si scartano, tenendo solo le ore che
    cadono nel giorno di calendario Europe/Rome richiesto."""
    import numpy as np

    times = series["time"].values
    values = series.values
    out = []
    for t, v in zip(times, values):
        if np.isnan(v):
            continue
        dt_utc = datetime.datetime.fromtimestamp(
            t.astype("datetime64[s]").astype(int), tz=datetime.timezone.utc
        )
        dt_local = dt_utc.astimezone(ROME_TZ)
        if dt_local.date() != target_date:
            continue
        out.append({"time": dt_local.strftime("%H:%M"), "level": round(float(v), 2)})
    return out


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
