```
docker compose up -d --build
docker exec -it the-barn-challenge bash
catkin_make && cd src/the-barn-challenge/
python run.py --gui --world_idx 108
rviz -d src/the-barn-challenge/jackal_nav.rviz 
./test.sh
./test_dynamic.sh
python report_test.py --out_path res/dwa_out.txt
```
