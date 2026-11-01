#include <logging.hpp>

#include <ios>

std::ofstream s_nn_log;

void set_nn_log_file(const std::string& filename, bool append)
{
    if (s_nn_log.is_open())
        s_nn_log.close();

    const auto mode = std::ios::out | (append ? std::ios::app : std::ios::trunc);
    s_nn_log.open(filename, mode);
}
