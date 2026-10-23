#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

#include "cyber/record/record_message.h"
#include "cyber/record/record_reader.h"

namespace {

struct TopicStats {
  uint64_t count = 0;
  uint64_t min_timestamp = std::numeric_limits<uint64_t>::max();
  uint64_t max_timestamp = 0;
  uint64_t hash = 14695981039346656037ULL;
};

void UpdateHash(uint64_t *hash, const void *data, size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < size; ++i) {
    *hash ^= bytes[i];
    *hash *= 1099511628211ULL;
  }
}

void UpdateHash(TopicStats *stats,
                const apollo::cyber::record::RecordMessage &msg) {
  for (int i = 0; i < 8; ++i) {
    const unsigned char byte = static_cast<unsigned char>(msg.time >> (i * 8));
    UpdateHash(&stats->hash, &byte, sizeof(byte));
  }
  UpdateHash(&stats->hash, msg.channel_name.data(), msg.channel_name.size());
  const unsigned char separator = 0;
  UpdateHash(&stats->hash, &separator, sizeof(separator));
  UpdateHash(&stats->hash, msg.content.data(), msg.content.size());
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: record_reader_probe RECORD [RECORD ...]\n";
    return 2;
  }

  const auto start = std::chrono::steady_clock::now();
  std::unordered_map<std::string, TopicStats> topics;
  uint64_t total_messages = 0;

  for (int arg = 1; arg < argc; ++arg) {
    apollo::cyber::record::RecordReader reader(argv[arg]);
    if (!reader.IsValid()) {
      std::cerr << "failed to open record: " << argv[arg] << '\n';
      return 1;
    }

    apollo::cyber::record::RecordMessage message;
    while (reader.ReadMessage(&message)) {
      ++total_messages;
      if (message.channel_name != "/apollo/sensor/gnss/imu" &&
          message.channel_name != "/apollo/sensor/vanjeelidar/up/PointCloud2") {
        continue;
      }
      auto &stats = topics[message.channel_name];
      ++stats.count;
      if (message.time < stats.min_timestamp) {
        stats.min_timestamp = message.time;
      }
      if (message.time > stats.max_timestamp) {
        stats.max_timestamp = message.time;
      }
      UpdateHash(&stats, message);
    }
  }

  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  std::cout << std::fixed << std::setprecision(3)
            << "elapsed_s=" << elapsed.count() << '\n'
            << "total_messages=" << total_messages << '\n';
  for (const auto &[topic, stats] : topics) {
    std::cout << "topic=" << topic << " count=" << stats.count
              << " min_timestamp_ns=" << stats.min_timestamp
              << " max_timestamp_ns=" << stats.max_timestamp
              << " hash=" << std::hex << std::setw(16) << std::setfill('0')
              << stats.hash << std::dec << '\n';
  }
  return 0;
}
