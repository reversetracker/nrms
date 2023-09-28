APP_NAME      := nrms
VERSION       := refactoring_v2
REGION        := ap-northeast-2
ECR           := 533448761297.dkr.ecr.$(REGION).amazonaws.com
DOCKER_REPO   := $(ECR)/$(APP_NAME):$(VERSION)

## Build the container
build:
	@echo "=> Building $(APP_NAME):$(VERSION)"
	docker buildx build --platform=linux/amd64 -t $(APP_NAME):$(VERSION) -f Dockerfile .

## Tag the container
tag:
	@echo "=> Tagging $(APP_NAME):$(VERSION) as $(DOCKER_REPO)"
	docker tag $(APP_NAME):$(VERSION) $(DOCKER_REPO)

## Run container on port 80
run:
	@echo "=> Running $(APP_NAME):$(VERSION) on port 80"
	docker run -it -p 80:80 $(APP_NAME):$(VERSION)

## Login to AWS Account
login-aws:
	@echo "=> Logging into AWS Account"
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR)

## Publish the `{version}` tagged container to ECR
push-aws:
	@echo "=> Publishing $(APP_NAME):$(VERSION) to $(DOCKER_REPO)"
	docker push $(DOCKER_REPO)

## Run build, tag, login-aws, and push-aws in sequence
release: build tag login-aws push-aws
