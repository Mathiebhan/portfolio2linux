#include <iostream>
#include <string>
#include <fstream>
#include <rw/rw.hpp>
#include <rw/kinematics/Kinematics.hpp>
#include <rwlibs/pathplanners/rrt/RRTPlanner.hpp>
#include <rwlibs/pathplanners/rrt/RRTQToQPlanner.hpp>
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>

using namespace std;
using namespace rw::common;
using namespace rw::math;
using namespace rw::kinematics;
using namespace rw::loaders;
using namespace rw::models;
using namespace rw::pathplanning;
using namespace rw::proximity;
using namespace rw::trajectory;
using namespace rwlibs::pathplanners;
using namespace rwlibs::proximitystrategies;

#define MAXTIME 10.

bool checkCollisions(Device::Ptr device, const State &state, const CollisionDetector &detector, const Q &q) {
    State testState;
    CollisionDetector::QueryResult data;
    bool colFrom;

    testState = state;
    device->setQ(q,testState);
    colFrom = detector.inCollision(testState,&data);
    if (colFrom) {
        cerr << "Configuration in collision: " << q << endl;
        cerr << "Colliding frames: " << endl;
        FramePairSet fps = data.collidingFrames;
        for (FramePairSet::iterator it = fps.begin(); it != fps.end(); it++) {
            cerr << (*it).first->getName() << " " << (*it).second->getName() << endl;
        }
        return false;
    }
    return true;
}

void writeToLUA(QPath path){
    ofstream myfile;
    myfile.open ("LUAscript.txt");
    cout<<"Writing to a LUA script."<<endl;
    myfile << "wc = rws.getRobWorkStudio():getWorkCell()\nstate = wc:getDefaultState()\ndevice = wc:findDevice(\"KukaKr16\")\ngripper = wc:findFrame(\"Tool\");\nbottle = wc:findFrame(\"Bottle\");\ntable = wc:findFrame(\"Table\");\n\n";
    myfile << "function setQ(q)\nqq = rw.Q(#q,q[1],q[2],q[3],q[4],q[5],q[6])\ndevice:setQ(qq,state)\nrws.getRobWorkStudio():setState(state)\nrw.sleep(0.1)\nend\n\n";
    myfile << "function attach(obj, tool)\nrw.gripFrame(obj, tool, state)\nrws.getRobWorkStudio():setState(state)\nrw.sleep(0.1)\nend\n\n";

    for (QPath::iterator it = path.begin(); it < path.end(); it++) {
        vector<double> joint_values;
        Q configuration=*it;
        configuration.Q::toStdVector(joint_values);
        myfile << "setQ({"<<joint_values[0]<<","<<joint_values[1]<<","<<joint_values[2]<<","<<joint_values[3]<<","<<joint_values[4]<<","<<joint_values[5]<<"})\n";
        if (it==path.begin()) myfile << "attach(bottle,gripper)\n";
    }

    myfile << "attach(bottle,table)\n";
    myfile.close();

}
int main(int argc, char** argv) {
    int value=1; // used for exporting data for MATLAB use, leave at 1 for normal program operation
    if(value==1){
        const string wcFile = "/home/student/Downloads/Kr16WallWorkCell/Scene.wc.xml";
        const string deviceName = "KukaKr16";
        cout << "Trying to use workcell " << wcFile << " and device " << deviceName << endl;

        WorkCell::Ptr wc = WorkCellLoader::Factory::load(wcFile);
        Device::Ptr device = wc->findDevice(deviceName);
        if (device == NULL) {
            cerr << "Device: " << deviceName << " not found!" << endl;
            return 0;
        }
        State state = wc->getDefaultState();
        const Q initial(6,-3.142,-0.827,-3.002,-3.143,0.099,-1.573);
        device->setQ(initial, state);
        const string gripperFrameName = "Tool";
        Frame* tool = wc->findFrame(gripperFrameName);
        if (tool == NULL) {
            cerr << "Frame: " << gripperFrameName << " not found!" << endl;
            return 0;
        }
        const string bottleFrameName = "Bottle";
        Frame* bottle = wc->findFrame(bottleFrameName);
        if (bottle == NULL) {
            cerr << "Frame: " << bottleFrameName << " not found!" << endl;
            return 0;
        }
        Kinematics::gripFrame(bottle,tool,state);

        Math::seed();

        CollisionDetector detector(wc, ProximityStrategyFactory::makeDefaultCollisionStrategy());
        PlannerConstraint constraint = PlannerConstraint::make(&detector,device,state);

        QSampler::Ptr sampler = QSampler::makeConstrained(QSampler::makeUniform(device),constraint.getQConstraintPtr());
        QMetric::Ptr metric = MetricFactory::makeEuclidean<Q>();
        double extend = 0.5;
        QToQPlanner::Ptr planner = RRTPlanner::makeQToQPlanner(constraint, sampler, metric, extend, RRTPlanner::RRTConnect);

        Q from(6,-3.142,-0.827,-3.002,-3.143,0.099,-1.573);
        Q to(6,1.571,0.006,0.030,0.153,0.762,4.490);

        if (!checkCollisions(device, state, detector, from))
            return 0;
        if (!checkCollisions(device, state, detector, to))
            return 0;

        cout << "Planning from " << from << " to " << to << endl;
        QPath path;
        Timer t;
        t.resetAndResume();
        planner->query(from,to,path,MAXTIME);
        t.pause();
        if (t.getTime() >= MAXTIME) {
            cout << "Notice: max time of " << MAXTIME << " seconds reached." << endl;
        }

        for (QPath::iterator it = path.begin(); it < path.end(); it++) {
            cout << *it << endl;
        }
        writeToLUA(path);
        cout << "Path of length " << path.size() << " found in " << t.getTime() << " seconds." << endl;

        cout << "Program done." << endl;
    }


    else if(value==2){ // We use this if clause only for exporting data for optimal path analysis to MATLAB, main code is in (value==1) clause
        ofstream datafile;
        datafile.open ("Optimal_path_data.txt");
        for(double extend=0.1; extend<=10; extend=extend+0.1)
        {
            const string wcFile = "/home/student/Downloads/Kr16WallWorkCell/Scene.wc.xml";
            const string deviceName = "KukaKr16";

            WorkCell::Ptr wc = WorkCellLoader::Factory::load(wcFile);
            Device::Ptr device = wc->findDevice(deviceName);
            if (device == NULL) {
                cerr << "Device: " << deviceName << " not found!" << endl;
                return 0;
            }
            State state = wc->getDefaultState();
            const Q initial(6,-3.142,-0.827,-3.002,-3.143,0.099,-1.573);
            device->setQ(initial, state);
            const string gripperFrameName = "Tool";
            Frame* tool = wc->findFrame(gripperFrameName);
            if (tool == NULL) {
                cerr << "Frame: " << gripperFrameName << " not found!" << endl;
                return 0;
            }
            const string bottleFrameName = "Bottle";
            Frame* bottle = wc->findFrame(bottleFrameName);
            if (bottle == NULL) {
                cerr << "Frame: " << bottleFrameName << " not found!" << endl;
                return 0;
            }
            Kinematics::gripFrame(bottle,tool,state);

            Math::seed();

            CollisionDetector detector(wc, ProximityStrategyFactory::makeDefaultCollisionStrategy());
            PlannerConstraint constraint = PlannerConstraint::make(&detector,device,state);

            QSampler::Ptr sampler = QSampler::makeConstrained(QSampler::makeUniform(device),constraint.getQConstraintPtr());
            QMetric::Ptr metric = MetricFactory::makeEuclidean<Q>();

            QToQPlanner::Ptr planner = RRTPlanner::makeQToQPlanner(constraint, sampler, metric, extend, RRTPlanner::RRTConnect);

            Q from(6,-3.142,-0.827,-3.002,-3.143,0.099,-1.573);
            Q to(6,1.571,0.006,0.030,0.153,0.762,4.490);

            if (!checkCollisions(device, state, detector, from))
                return 0;
            if (!checkCollisions(device, state, detector, to))
                return 0;
            QPath path;
            Timer t;
            t.resetAndResume();
            planner->query(from,to,path,MAXTIME);
            t.pause();
            if (t.getTime() >= MAXTIME) {
                cout << "Notice: max time of " << MAXTIME << " seconds reached." << endl;
            }

            //            for (QPath::iterator it = path.begin(); it < path.end(); it++) {
            //                cout << *it << endl;
            //            }

            cout << "Path of length " << path.size() << " found in " << t.getTime() << " seconds for extend value of " <<extend<< endl;

            datafile << extend << " "<< path.size() << " "<< t.getTime() << "\n";
        }
        datafile.close();

        cout << "Done exporting data for optimal path analysis." << endl;

    }
    else if(value==3){ // We use this if clause only for exporting data for repeatability to MATLAB, main code is in (value==1) clause
        ofstream datafile;
        datafile.open ("Repeatability_data.txt");
        for(int i=0; i<100; i++){
            const string wcFile = "/home/student/Downloads/Kr16WallWorkCell/Scene.wc.xml";
            const string deviceName = "KukaKr16";

            WorkCell::Ptr wc = WorkCellLoader::Factory::load(wcFile);
            Device::Ptr device = wc->findDevice(deviceName);
            if (device == NULL) {
                cerr << "Device: " << deviceName << " not found!" << endl;
                return 0;
            }
            State state = wc->getDefaultState();
            const Q initial(6,-3.142,-0.827,-3.002,-3.143,0.099,-1.573);
            device->setQ(initial, state);
            const string gripperFrameName = "Tool";
            Frame* tool = wc->findFrame(gripperFrameName);
            if (tool == NULL) {
                cerr << "Frame: " << gripperFrameName << " not found!" << endl;
                return 0;
            }
            const string bottleFrameName = "Bottle";
            Frame* bottle = wc->findFrame(bottleFrameName);
            if (bottle == NULL) {
                cerr << "Frame: " << bottleFrameName << " not found!" << endl;
                return 0;
            }
            Kinematics::gripFrame(bottle,tool,state);

            Math::seed();

            CollisionDetector detector(wc, ProximityStrategyFactory::makeDefaultCollisionStrategy());
            PlannerConstraint constraint = PlannerConstraint::make(&detector,device,state);

            QSampler::Ptr sampler = QSampler::makeConstrained(QSampler::makeUniform(device),constraint.getQConstraintPtr());
            QMetric::Ptr metric = MetricFactory::makeEuclidean<Q>();
            double extend = 0.7;
            QToQPlanner::Ptr planner = RRTPlanner::makeQToQPlanner(constraint, sampler, metric, extend, RRTPlanner::RRTConnect);

            Q from(6,-3.142,-0.827,-3.002,-3.143,0.099,-1.573);
            Q to(6,1.571,0.006,0.030,0.153,0.762,4.490);

            if (!checkCollisions(device, state, detector, from))
                return 0;
            if (!checkCollisions(device, state, detector, to))
                return 0;

            QPath path;
            Timer t;
            t.resetAndResume();
            planner->query(from,to,path,MAXTIME);
            t.pause();
            if (t.getTime() >= MAXTIME) {
                cout << "Notice: max time of " << MAXTIME << " seconds reached." << endl;
            }

            cout << "Path of length " << path.size() << " found in " << t.getTime() << " seconds for extend value of " <<extend<< endl;
            datafile << extend << " "<< path.size() << " "<< t.getTime() << "\n";

        }
        datafile.close();

        cout << "Done exporting data for repeatability analysis." << endl;
    }
    return 0;
}
