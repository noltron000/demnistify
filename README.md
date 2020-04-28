# Steps
1. build local contents to image
	`docker build ./`
1. tag the local image
	`docker tag 123456abcdef`
1. tag the local & remote images
	`docker tag local-image:tagname username/reponame:tagname`
1. `docker image tag 2ec69fa4b332 mnist`
##
Push a repo:
- `docker push username/reponame:tagname`
- `docker push noltron000/flask_app:latest`
