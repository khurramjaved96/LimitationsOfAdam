cmake_minimum_required(VERSION 3.14)
project(LimitationsOfAdam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(WalkingSitting walking_sitting_experiment.cpp include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
         src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp)

add_executable(IDBDPaper idbd_paper_experiment.cpp include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
         src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp)

add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp)

target_link_libraries(WalkingSitting PUBLIC mariadb)
target_link_libraries(WalkingSitting PUBLIC tbb)

target_link_libraries(IDBDPaper PUBLIC mariadb)
target_link_libraries(IDBDPaper PUBLIC tbb)

target_link_libraries(Count PUBLIC mariadb)
target_link_libraries(Count PUBLIC tbb)

project(LimitationsOfAdam LANGUAGES CXX)

