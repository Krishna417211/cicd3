import mlflow
from mlflow.tracking import MlflowClient



mlflow.set_tracking_uri('sqlite:///mlflow.db')   # same db train.py wrote
MODEL_NAME = "taxi_model"
METRIC = "mae"
LOWER_IS_BETTER = True
client = MlflowClient()
...
client = MlflowClient()

# latest registered version (any stage), by version number
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise SystemExit(f"No versions found for '{MODEL_NAME}'")
new = max(versions, key=lambda v: int(v.version))
print(f"Latest version: v{new.version}")

def metric_for(version_obj):
    run = client.get_run(version_obj.run_id)
    return run.data.metrics.get(METRIC)

new_metric = metric_for(new)
if new_metric is None:
    raise SystemExit(f"Metric '{METRIC}' not found on run for v{new.version}")
print(f"New v{new.version} {METRIC}={new_metric}")

prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
current = prod[0] if prod else None

def better(a, b):
    if b is None:
        return True
    return a < b if LOWER_IS_BETTER else a > b

if current is None:
    print("No current Production model.")
    promote = True
else:
    cur_metric = metric_for(current)
    print(f"Current Production v{current.version} {METRIC}={cur_metric}")
    promote = better(new_metric, cur_metric)

if promote:
    client.transition_model_version_stage(
        MODEL_NAME, new.version, stage="Production",
        archive_existing_versions=True,
    )
    print(f"DEPLOY: promoted v{new.version} to Production.")
else:
    client.transition_model_version_stage(
        MODEL_NAME, new.version, stage="Staging",
    )
    print(f"ROLLBACK: v{new.version} did not beat Production "
          f"v{current.version}. Kept in Staging.")