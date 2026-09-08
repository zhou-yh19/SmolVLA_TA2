rsync -avz --progress \
	--exclude='.git/' \
	--exclude='weights' \
	--exclude='weights/' \
	--exclude='datasets' \
	--exclude='datasets/' \
	--exclude='outputs/' \
	--exclude='outputs' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    	./ \
  	/home/party/Documents/ICRA_robotics/SmolVLA_TA2/ \
  	arapat@120.92.116.147:/home/arapat/disk0/SmolVLA_TA2/
