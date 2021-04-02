#ifndef IMPLICIT_GPU_DEVICE_BUFFER_H_
#define IMPLICIT_GPU_DEVICE_BUFFER_H_

namespace implicit { namespace gpu {

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size);
    DeviceBuffer(size_t size, const T * elements);
    ~DeviceBuffer();

    const T * get() const { return data; }
    T * get() { return data; }
    size_t size() const { return size_; }

    DeviceBuffer(DeviceBuffer<T> && other);

    DeviceBuffer(const DeviceBuffer<T> &) = delete;
    DeviceBuffer<T> & operator=(const DeviceBuffer<T> &) = delete;

protected:
    size_t size_;
    T * data;
};
}}  // namespace implicit/gpu
#endif  // IMPLICIT_GPU_DEVICE_BUFFER_H_
