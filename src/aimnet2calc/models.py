import requests
import sys

from pathlib import Path

model_registry_aliases = {
    'aimnet2'       : 'aimnet2_wb97m_0',
    'aimnet2_wb97m' : 'aimnet2_wb97m_0',
    'aimnet2_b973c' : 'aimnet2_b973c_0',
    'aimnet2-qr'    : 'aimnet2-qr_b97md4_qzvp_2',
}

download_url = "https://github.com/zubatyuk/aimnet-model-zoo/raw/main/aimnet2/"

def get_model_path(name: str) -> str:
    assets_path = Path(__file__).parent / "assets"
    assets_path.mkdir(exist_ok=True)

    if name in model_registry_aliases:
        model_path = assets_path / f"{model_registry_aliases[name]}.jpt"
        if model_path.exists():
            pass
        else:
            url = download_url + f"{model_registry_aliases[name]}.jpt"
            print(f"Downloading model file from {url}")
            r = requests.get(url)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(r.content)
            print(f"Saved to {model_path}")
        return model_path.as_posix()
    else:
        supported_models = [k for k in model_registry_aliases]
        print(f"requested model {s} is not supported.")
        print("supported models= " + ','.join(supported_models))
        sys.exit(0)