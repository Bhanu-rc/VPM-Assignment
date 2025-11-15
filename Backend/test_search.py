import requests, json, sys

url = "http://127.0.0.1:5000/search"
path = "data/images/0.jpg"

try:
    files = {"image": open(path, "rb")}
    r = requests.post(url, files=files, timeout=30)
    print("STATUS:", r.status_code)
    ct = r.headers.get("content-type", "")
    print("CONTENT-TYPE:", ct)
    if "application/json" in ct:
        print(json.dumps(r.json(), indent=2))
    else:
        print(r.text[:2000])
except Exception as e:
    print("ERROR:", e)
    sys.exit(1)
