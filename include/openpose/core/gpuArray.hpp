#ifndef OPENPOSE__CORE__GPU_ARRAY_HPP
#define OPENPOSE__CORE__GPU_ARRAY_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <boost/shared_ptr.hpp> // Note: std::shared_ptr not (fully) supported for array pointers: http://stackoverflow.com/questions/8947579/

#include "openpose/config.hpp"

namespace op
{
    /**
     * Array<T>: The OpenPose Basic Raw Data Container
     * This template class implements a multidimensional data array. It is our basic data container, analogous to cv::Mat in OpenCV, Tensor in
     * Torch/TensorFlow or Blob in Caffe.
     * It wraps a cv::Mat and a boost::shared_ptr, both of them pointing to the same raw data. I.e. they both share the same memory, so we can read
     * and modify this data in both formats with no performance impact.
     * Hence, it keeps high performance while adding high-level functions.
     */
    template<typename T>
    class OPENPOSE_API GpuArray
    {
    public:
        // -------------------------------------------------- Constructors and Data Allocator Functions -------------------------------------------------- //
        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const int size).
         * @param size Integer with the number of T element to be allocated. E.g. size = 5 is internally similar to: new T[5].
         */
        explicit GpuArray(const int size);

        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const std::vector<int>& size = {}).
         * @param sizes Vector with the size of each dimension. E.g. size = {3, 5, 2} is internally similar to: new T[3*5*2].
         */
        explicit GpuArray(const std::vector<int>& sizes = {});

        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const int size, const T value).
         * @param size Integer with the number of T element to be allocated. E.g. size = 5 is internally similar to: new T[5].
         * @param value Initial value for each component of the Array.
         */
        explicit GpuArray(const int size, const T value);

        /**
         * Array constructor.
         * Equivalent to default constructor + reset(const std::vector<int>& size, const T value).
         * @param sizes Vector with the size of each dimension. E.g. size = {3, 5, 2} is internally similar to: new T[3*5*2].
         * @param value Initial value for each component of the Array.
         */
        explicit GpuArray(const std::vector<int>& sizes, const T value);

        /**
         * Copy constructor.
         * It performs `fast copy`: For performance purpose, copying a Array<T> or Datum or cv::Mat just copies the reference, it still shares the same internal data.
         * Modifying the copied element will modify the original one.
         * Use clone() for a slower but real copy, similarly to cv::Mat and Array<T>.
         * @param array Array to be copied.
         */
		GpuArray<T>(const GpuArray<T>& array);

        /**
         * Copy assignment.
         * Similar to Array<T>(const Array<T>& array).
         * @param array Array to be copied.
         * @return The resulting Array.
         */
		GpuArray<T>& operator=(const GpuArray<T>& array);

        /**
         * Move constructor.
         * It destroys the original Array to be moved.
         * @param array Array to be moved.
         */
		GpuArray<T>(GpuArray<T>&& array);

        /**
         * Move assignment.
         * Similar to Array<T>(Array<T>&& array).
         * @param array Array to be moved.
         * @return The resulting Array.
         */
		GpuArray<T>& operator=(GpuArray<T>&& array);

        /**
         * Clone function.
         * Similar to cv::Mat::clone and Datum::clone.
         * It performs a real but slow copy of the data, i.e., even if the copied element is modified, the original one is not.
         * @return The resulting Array.
         */
		GpuArray<T> clone() const;

		void copyTo(GpuArray<T>& dst) const;

        /**
         * Data allocation function.
         * It allocates the required space for the memory (it does not initialize that memory).
         * @param size Integer with the number of T element to be allocated. E.g. size = 5 is internally similar to: new T[5].
         */
        void reset(const int size);

        /**
         * Data allocation function.
         * Similar to reset(const int size), but it allocates a multi-dimensional array of dimensions each of the values of the argument.
         * @param sizes Vector with the size of each dimension. E.g. size = {3, 5, 2} is internally similar to: new T[3*5*2].
         */
        void reset(const std::vector<int>& sizes = {});

        /**
         * Data allocation function.
         * Similar to reset(const int size), but initializing the data to the value specified by the second argument.
         * @param size Integer with the number of T element to be allocated. E.g. size = 5 is internally similar to: new T[5].
         * @param value Initial value for each component of the Array.
         */
        void reset(const int size, const T value);

        /**
         * Data allocation function.
         * Similar to reset(const std::vector<int>& size), but initializing the data to the value specified by the second argument.
         * @param sizes Vector with the size of each dimension. E.g. size = {3, 5, 2} is internally similar to: new T[3*5*2].
         * @param value Initial value for each component of the Array.
         */
        void reset(const std::vector<int>& sizes, const T value);

       
		/**
		* Data allocation function.
		* It internally assigns all the allocated memory to the value indicated by the argument.
		* @param value Value for each component of the Array.
		*/
		void setTo(const T value);
       
        // -------------------------------------------------- Data Information Functions -------------------------------------------------- //
        /**
         * Check whether memory has been allocated.
         * @return True if no memory has been allocated, false otherwise.
         */
        inline bool empty() const
        {
            return (mVolume == 0);
        }

        /**
         * Return a vector with the size of each dimension allocated.
         * @return A std::vector<int> with the size of each dimension. If no memory has been allocated, it will return an empty std::vector.
         */
        inline std::vector<int> getSize() const
        {
            return mSize;
        }

        /**
         * Return a vector with the size of the desired dimension.
         * @param index Dimension to check its size.
         * @return Size of the desired dimension. It will return 0 if the requested dimension is higher than the number of dimensions.
         */
        int getSize(const int index) const;

        /**
         * Return the total number of dimensions, equivalent to getSize().size().
         * @return The number of dimensions. If no memory is allocated, it returns 0.
         */
        inline size_t getNumberDimensions() const
        {
            return mSize.size();
        }

        /**
         * Return the total number of elements allocated, equivalent to multiply all the components from getSize().
         * E.g. for a Array<T> of size = {2,5,3}, the volume or total number of elements is: 2x5x3 = 30.
         * @return The total volume of the allocated data. If no memory is allocated, it returns 0.
         */
        inline size_t getVolume() const
        {
            return mVolume;
        }

        /**
         * Similar to getVolume(), but in this case it just returns the volume between the desired dimensions.
         * E.g. for a Array<T> of size = {2,5,3}, the volume or total number of elements for getVolume(1,2) is: 5x3 = 15.
         * @return The total volume of the allocated data between the desired dimensions. If the index are out of bounds, it throws an error.
         */
        size_t getVolume(const int indexA, const int indexB) const;



        // -------------------------------------------------- Data Access Functions And Operators -------------------------------------------------- //
        /**
         * Return a raw pointer to the data. Similar to: boost::shared_ptr::get().
         * Note: if you modify the pointer data, you will directly modify it in the Array<T> instance too.
         * If you know you do not want to modify the data, then use getConstPtr() instead.
         * @return A raw pointer to the data.
         */
        inline T* getPtr()
        {
            return spData.get();
        }

        /**
         * Similar to getPtr(), but it forbids the data to be edited.
         * @return A raw const pointer to the data.
         */
        inline const T* getConstPtr() const
        {
            return spData.get();
        }

      

       
    private:
        std::vector<int> mSize;
        size_t mVolume;
        boost::shared_ptr<T> spData;
  
      
    };
}

#endif // OPENPOSE__CORE__ARRAY_HPP
