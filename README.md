# Video deblur with https://github.com/xinntao/EDVR deblur model.

## Install dependencies:
```
pip install -r requirements.txt
```

## Compile deformable convilution:
```
cd EDVR/codes/models/archs/dcn
python setup.py develop
```

## Run local flask server:
```
FLASK_ENV=development FLASK_APP=app.py flask run
```

## Send request containing video path:
curl -X POST -F file=@video.mp4 http://localhost:5000/process
