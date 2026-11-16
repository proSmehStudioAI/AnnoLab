#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "httpServer.hpp"
#include "globals.hpp"
#include "loginHandler.hpp"
#include "projectsHandler.hpp"
#include "createProjectHandler.hpp"
#include "imageNavigationHandler.hpp"
#include "getClassesHandler.hpp"
#include "saveAnnotationsHandler.hpp"
#include "getAnnotationsHandler.hpp"
#include "uploadImageHandler.hpp"
#include "downloadDatasetHandler.hpp"
#include "downloadModelHandler.hpp"
#include "setProjectStateHandler.hpp"
#include "updateTrainParamsHandler.hpp"
#include "getProjectInfoHandler.hpp"
#include "autodetectHandler.hpp"
#include "TrainingProcessManager.hpp"
#include "deleteProjectHandler.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

std::vector<std::pair<std::string, std::string>> loadPasswordTxt(const std::string& path)
{
    std::vector<std::pair<std::string, std::string>> accaunt;

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Не удалось открыть файл: " << path << '\n';
        return accaunt;
    }

    std::string login, password;
    while (file >> login >> password)
        accaunt.emplace_back(login, password);

    return accaunt;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    globals::accaunt = loadPasswordTxt("password.txt");

    std::string worker = (argc > 2) ? argv[2]
        : "pythonWorkerTrain\\pythonWorkerTrainYolov.py";
    {
        std::error_code ec;
        auto abs = std::filesystem::absolute(worker, ec).lexically_normal();
        std::cout << "[main] worker script: " << abs.string()
            << "  exists="
            << (std::filesystem::is_regular_file(abs, ec) ? "yes" : "no")
            << "\n";
    }

    auto& mgr = training_manager::Manager::instance();
    mgr.setPythonCommand(argc > 1 ? argv[1] : "py");
    mgr.setWorkerScript(worker);
    mgr.setEpochsTotal(1000);

    httpHttpsServer serv("127.0.0.1", 80, 443, "server.key", "server.crt");

    serv.addApiHandler("isLogin", &login_event::handle);
    serv.addApiHandler("checkprojectsstate", &projects_event::handle);
    serv.addApiHandler("createproject", &create_project_event::handle);
    serv.addApiHandler("imagenav", &image_nav_event::handle);
    serv.addApiHandler("rawimage", &image_nav_event::handleRaw);
    serv.addApiHandler("getclasses", &get_classes_event::handle);
    serv.addApiHandler("saveannotations", &save_annotations_event::handle);
    serv.addApiHandler("getannotations", &get_annotations_event::handle);
    serv.addApiHandler("uploadimage", &upload_image_event::handle);
    serv.addApiHandler("downloaddataset", &download_dataset_event::handle);
    serv.addApiHandler("downloadmodel", &download_model_event::handle);
    serv.addApiHandler("getprojectinfo", &get_project_info_event::handle);
    serv.addApiHandler("updatetrainparams", &update_train_params_event::handle);
    serv.addApiHandler("setprojectstate", &set_project_state_event::handle);
    serv.addApiHandler("autodetect", &autodetect_event::handle);
    serv.addApiHandler("deleteproject", &delete_project_event::handle);

    {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::is_directory("projects", ec)) {
            for (auto it = fs::directory_iterator("projects", ec);
                it != fs::directory_iterator(); it.increment(ec)) {
                if (ec) break;
                if (!it->is_directory()) continue;

                auto name = it->path().filename().string();

                if (project_files::readStatusBool(it->path())) {
                    mgr.start(name, it->path());
                }

                project_files::TrainSettings s;
                project_files::readSettings(it->path() / "modelSettings.json", s);
                std::error_code mEc;
                bool hasModel = fs::is_regular_file(it->path() / (s.modelSelected + "_best.pt"), mEc);
                if (!hasModel) hasModel = fs::is_regular_file(it->path() / (s.modelSelected + "_last.pt"), mEc);
                if (!hasModel) continue;

                fs::path warmup = it->path() / "__warmup__.jpg";
                std::thread([name, warmup]() {
                    training_manager::Manager::instance()
                        .runInference(name, warmup, 120000);
                    }).detach();
            }
        }
    }

    std::atomic<bool> pollerRunning{ true };
    std::thread poller([&] {
        while (pollerRunning.load()) {
            mgr.poll();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        });

    serv.run();

    pollerRunning.store(false);
    if (poller.joinable()) poller.join();

    training_manager::Manager::instance().shutdownAllInference();

    return 0;
}