/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    return rand() % end;
}

vector<int> del_ele_from_vector(vector<int> iVec, int del_ele)
{
    // for(vector<int>::iterator it=iVec.begin();it!=iVec.end();++it)
    // {
    //     if(*it == del_ele)
    //     {
    //         it=iVec.erase(it);
    //     }
    //     else
    //         ++it;
    // }

    return iVec;
}

std::tuple<py::array_t<int>, py::array_t<int>> padding_dict_to_ndarray_test_cpp(map<int,std::vector<int>> dict_name, int n_cols, int max_len, int mask_ele)
{
    py::array_t<int> ndarray = py::array_t<int>({n_cols, max_len});
    py::buffer_info buf1 = ndarray.request();
    int *ptr_neibor = (int *)buf1.ptr;
    py::array_t<int> n_neibors_index = py::array_t<int>({n_cols, 1});
    py::buffer_info buf2 = n_neibors_index.request();
    int *ptr_n_neibor = (int *)buf2.ptr;

    for (map<int,std::vector<int>>::iterator it=dict_name.begin();it!=dict_name.end();it++)
    {
        int uid = it->first;
        std::vector<int> neibor = it->second;
        if( max_len < neibor.size()) 
        {
            int n_neibor = max_len;
            random_shuffle(neibor.begin(), neibor.end());
            neibor.resize(max_len);
            ptr_n_neibor[uid] = n_neibor;
            for (int j=0; j < max_len; j++)
            {
                ptr_neibor[uid*max_len+j] = neibor.at(j);
            }
        }
        else
        {
            int n_neibor = neibor.size();
            neibor.resize( max_len, mask_ele );
            ptr_n_neibor[uid] = n_neibor;
            for (int j=0; j < max_len; j++)
            {
                ptr_neibor[uid*max_len+j] = neibor.at(j);
            }
        }
    }
    std::tuple<py::array_t<int>, py::array_t<int>> result(ndarray, n_neibors_index);
    return result;
}

std::tuple<py::array_t<int>, py::array_t<int>> padding_dict_to_ndarray_delpos_cpp(map<int,std::vector<int>> dict_name, vector<int> users, vector<int> items, int max_len, int mask_ele, int is_del_pos)
{
    int n_interactions = users.size();
    // cout << "n_interactions: " << n_interactions << endl;
    py::array_t<int> ndarray = py::array_t<int>({n_interactions, max_len});
    py::buffer_info buf1 = ndarray.request();
    int *ptr_neibor = (int *)buf1.ptr;
    py::array_t<int> n_neibors_index = py::array_t<int>({n_interactions, 1});
    py::buffer_info buf2 = n_neibors_index.request();
    int *ptr_n_neibor = (int *)buf2.ptr;

    for (int i=0; i<n_interactions; i++)
    {
        int uid = users.at(i);
        int iid = items.at(i);
        std::vector<int> neibor = dict_name[uid];
        if(is_del_pos){
            for(vector<int>::iterator it=neibor.begin(); it!=neibor.end(); it++)
            {
                if(*it == iid)
                {
                    neibor.erase(it);
                    break;
                }
            }
        }
        
        if( max_len < neibor.size()) 
        {
            int n_neibor = max_len;
            random_shuffle(neibor.begin(), neibor.end());
            neibor.resize(max_len);
            // cout << "uid:" << uid << endl;
            ptr_n_neibor[i] = n_neibor;
            for (int j=0; j < max_len; j++)
            {
                ptr_neibor[i*max_len+j] = neibor.at(j);
            }
        }
        else
        {
            int n_neibor = neibor.size();
            neibor.resize( max_len, mask_ele );
            // cout << "uid: " << uid << endl;
            ptr_n_neibor[i] = n_neibor;
            for (int j=0; j < max_len; j++)
            {
                ptr_neibor[i*max_len+j] = neibor.at(j);
            }
        }
    }
    // cout << "del finished" << endl;
    std::tuple<py::array_t<int>, py::array_t<int>> result(ndarray, n_neibors_index);
    return result;
}

py::array_t<int> get_0_from_tuple(std::tuple<py::array_t<int>, py::array_t<int>> result)
{
    return std::get<0>(result);
}

py::array_t<int> get_1_from_tuple(std::tuple<py::array_t<int>, py::array_t<int>> result)
{
    return std::get<1>(result);
}

py::array_t<int> pad_sequence_cpp(py::array_t<int> n_neibors_index, int max_len)
{
    int n_interactions = n_neibors_index.size();
    py::array_t<int> ndarray = py::array_t<int>({n_interactions, max_len});
    py::buffer_info buf1 = ndarray.request();
    int *ptr_neibor = (int *)buf1.ptr;
    py::buffer_info buf2 = n_neibors_index.request();
    int *ptr_index = (int *)buf2.ptr;
    for ( int i = 0; i < n_neibors_index.size(); i++)
    {
        vector<int> neibor(ptr_index[i], 1);
        neibor.resize(max_len, 0);
        for (int j=0; j< max_len; j++)
        {
            ptr_neibor[i*max_len + j] = neibor.at(j);
        }
    }
    return ndarray;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(padding, m)
{
    srand(time(0));
    // srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("padding_dict_to_ndarray_test_cpp", &padding_dict_to_ndarray_test_cpp, "padding",
          "dict_name"_a, "n_cols"_a, "max_len"_a, "mask_ele"_a);
    m.def("padding_dict_to_ndarray_delpos_cpp", &padding_dict_to_ndarray_delpos_cpp, "padding",
          "dict_name"_a, "users"_a, "items"_a, "max_len"_a, "mask_ele"_a, "is_del_pos"_a);
    m.def("get_0_from_tuple", &get_0_from_tuple, "get ele", "result"_a);
    m.def("get_1_from_tuple", &get_1_from_tuple, "get ele", "result"_a);
    m.def("del_ele_from_vector", &del_ele_from_vector, "del_ele", "iVec"_a, "del_ele"_a);
    m.def("pad_sequence_cpp", &pad_sequence_cpp, "pad sequence", "n_neibors_index"_a, "max_len"_a);
}