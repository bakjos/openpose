#include <typeinfo> // typeid
#include <numeric> // std::accumulate
#include "openpose/utilities/errorAndLog.hpp"
#include "openpose/core/gpuArray.hpp"
#include "openpose/utilities/cuda.hpp"

namespace op
{

	template <class T>
	static void DeallocateGpu(T* mem)
	{
		cudaFree(mem);
		cudaCheck(__LINE__, __FUNCTION__, __FILE__);

	}

    template<typename T>
    GpuArray<T>::GpuArray(const int size)
    {
        try
        {
            reset(size);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    GpuArray<T>::GpuArray(const std::vector<int>& sizes)
    {
        try
        {
            reset(sizes);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    GpuArray<T>::GpuArray(const int size, const T value)
    {
        try
        {
            reset(size, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    GpuArray<T>::GpuArray(const std::vector<int>& sizes, const T value)
    {
        try
        {
            reset(sizes, value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    GpuArray<T>::GpuArray(const GpuArray<T>& GpuArray) :
        mSize{GpuArray.mSize},
        mVolume{GpuArray.mVolume},
        spData{GpuArray.spData}
    {
    }

    template<typename T>
    GpuArray<T>& GpuArray<T>::operator=(const GpuArray<T>& GpuArray)
    {
        try
        {
            mSize = GpuArray.mSize;
            mVolume = GpuArray.mVolume;
            spData = GpuArray.spData;
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    GpuArray<T>::GpuArray(GpuArray<T>&& GpuArray) :
        mSize{GpuArray.mSize},
        mVolume{GpuArray.mVolume}
    {
        try
        {
            std::swap(spData, GpuArray.spData);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    GpuArray<T>& GpuArray<T>::operator=(GpuArray<T>&& GpuArray)
    {
        try
        {
            mSize = GpuArray.mSize;
            mVolume = GpuArray.mVolume;
            std::swap(spData, GpuArray.spData);
            // Return
            return *this;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return *this;
        }
    }

    template<typename T>
    GpuArray<T> GpuArray<T>::clone() const
    {
        try
        {
            // Constructor
            GpuArray<T> gpuArray{mSize};
			copyTo(gpuArray);
            // Return
            return std::move(gpuArray);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return GpuArray<T>{};
        }
    }

	template<typename T>
	void GpuArray<T>::copyTo(GpuArray<T>& dst) const {
		if ( mVolume != dst.mVolume) {
			dst.reset(mSize);
		}

		cudaMemcpy(dst.getPtr(), getConstPtr(), mVolume * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaCheck(__LINE__, __FUNCTION__, __FILE__);
	}

    template<typename T>
    void GpuArray<T>::reset(const int size)
    {
        try
        {
            if (size > 0)
                reset(std::vector<int>{size});
            else
                reset(std::vector<int>{});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void GpuArray<T>::reset(const std::vector<int>& sizes)
    {
        try
        {
            if (!sizes.empty())
            {
                // New size & volume
                mSize = sizes;
                mVolume = {std::accumulate(sizes.begin(), sizes.end(), 1ul, std::multiplies<size_t>())};
				T* data = nullptr;
				cudaMalloc((void**)&data, mVolume*sizeof(T));
				cudaCheck(__LINE__, __FUNCTION__, __FILE__);
				spData.reset(data, &DeallocateGpu<T>);
             
            }
            else
            {
                mSize = {};
                mVolume = 0;
                spData.reset();
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void GpuArray<T>::reset(const int sizes, const T value)
    {
        try
        {
            reset(sizes);
            setTo(value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename T>
    void GpuArray<T>::reset(const std::vector<int>& sizes, const T value)
    {
        try
        {
            reset(sizes);
            setTo(value);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

	template<typename T>
	void GpuArray<T>::setTo(const T value)
	{
		try
		{
			if (mVolume > 0){
				//TODO: Copy using cuda
			}
		}
		catch (const std::exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}
   

    template<typename T>
    int GpuArray<T>::getSize(const int index) const
    {
        try
        {
            // Matlab style:
                // If empty -> return 0
                // If index >= # dimensions -> return 1
            if (index < mSize.size() && 0 <= index)
                return mSize[index];
            // Long version:
            // else if (mSize.empty())
            //     return 0;
            // else // if mSize.size() <= index 
            //     return 1;
            // Equivalent to:
            else
                return (!mSize.empty());
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

    template<typename T>
    size_t GpuArray<T>::getVolume(const int indexA, const int indexB) const
    {
        try
        {
            if (indexA < indexB)
            {
                if (0 <= indexA && indexB < mSize.size()) // 0 <= indexA < indexB < mSize.size()
                    return std::accumulate(mSize.begin()+indexA, mSize.begin()+indexB+1, 1ul, std::multiplies<size_t>());
                else
                {
                    error("Indexes out of dimension.", __LINE__, __FUNCTION__, __FILE__);
                    return 0;
                }
            }
            else if (indexA == indexB)
                return mSize.at(indexA);
            else // if (indexA > indexB)
            {
                error("indexA > indexB.", __LINE__, __FUNCTION__, __FILE__);
                return 0;
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0;
        }
    }

   
    COMPILE_TEMPLATE_BASIC_TYPES(GpuArray);
}
