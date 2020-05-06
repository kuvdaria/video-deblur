# Video deblur with https://github.com/xinntao/EDVR deblur model.

# To run locally:

  ## Install dependencies:
  ```
  pip install -r requirements.txt
  ```

  ## Run local application:
  ```
  FLASK_ENV=development FLASK_APP=app.py flask run
  ```

  ## Send request containing video paths of input and output videos:
  ```
  curl -X POST -F file=@video.mp4 http://localhost:5000/process --output video_deblurred.mp4
  ```
  
# To run with Floydhub:
Create an account with ACCAUNT_NAME and a project with PROJECT_NAME, then

## Deploy with Floydhub
  ```
  pip install floyd-cli
  floyd login
  git clone https://github.com/kuvdaria/video-deblur
  floyd inint ACCOUNT_NAME/PROJECT_NAME
  floyd run --gpu --mode serve
  ```
  
 ## Send request:
 ```
 curl -o video_deblurred.mp4 -F "file=@video.mp4" https://www.floydlabs.com/serve/ACCOUNT_NAME/projects/PROJECT_NAME/process
 ```
 
 # Send request to my running application:
 ```
 curl -o video_deblurred.mp4 -F "file=@video.mp4" https://www.floydlabs.com/serve/kuvdaria/projects/video_deblur/process
 ```
 
 if upstream times out, you can get deblurred file by running
 ```
 wget https://www.floydhub.com/viewer/data/td95FTFusgsSmAtfKZ4vxJ/videos/video_done_aud.mp4
 ```


