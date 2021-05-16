// Copyright 2004-present Facebook. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

// PreprocessMesh.cpp processes each *.obj file separately and in parallel.

// extern claims an external function.
extern pangolin::GlSlProgram GetShaderProgram();

//This is the function for only vertices creating
void CreateVertices( pangolin::Geometry& geom,
    std::vector<Eigen::Vector3f>& surfpts,
    int num_sample){
 
  // It is probably fine to remain the vertices as 3D vectors.
  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);
}

void SampleFromSurface(
    pangolin::Geometry& geom,
    std::vector<Eigen::Vector3f>& surfpts,
    int num_sample) {
  float total_area = 0.0f;

  std::vector<float> cdf_by_area;

  std::vector<Eigen::Vector3i> linearized_faces;

  // According to the source code of Pangolin, ibo should refer to the int vertex_indices of geometric elements.
  // For triangles, the length of ibo is 3, while for lines, the length is 2.
  // It is probably fine to remain the vertices as 3D vectors.
  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {
      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

      for (int i = 0; i < ibo.h; ++i) {
        linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
      }
    }
  }

  // Vertices are here processed as images with float-value pixels.
  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

  // As inferred, a face, as an instance in linearized_faces, saves the indices of its vertices, which are named as face(i).
  // These indices are processed by the function RowPtr to query the corresponding vertices as 3D vectors.
  // These vertices are processed by the function TriangleArea to get the area.
  // In our 2D case, we should calculate the total perimeter.
  // TriangleArea is included in Utils.h.
  for (const Eigen::Vector3i& face : linearized_faces) {
    float area = TriangleArea(
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

    if (std::isnan(area)) {
      area = 0.f;
    }

    total_area += area;

    // cdf_by_area might refer to the accumulated density function of area.
    if (cdf_by_area.empty()) {
      cdf_by_area.push_back(area);

    } else {
      cdf_by_area.push_back(cdf_by_area.back() + area);
    }
  }

  std::random_device seeder;
  // mt19937 creates a pseudo random number generator.
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_area);

  // This operation sample points on triangles weighted by area.
  while ((int)surfpts.size() < num_sample) {
    float tri_sample = rand_dist(generator);
    std::vector<float>::iterator tri_index_iter =
        lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
      
    // The substraction of two iterators is a signed integer, which is required in next line.
    int tri_index = tri_index_iter - cdf_by_area.begin();

    const Eigen::Vector3i& face = linearized_faces[tri_index];

    // SamplePointFromTriangle is included in Utils.h.
    surfpts.push_back(SamplePointFromTriangle(
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2)))));
  }
}

// SampleSDFnearSurface refers to the variations of surface points. This can be inferred since a kdTree is input.
void SampleSDFNearSurface(
    KdVertexListTree& kdTree,
    std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& xyz_surf,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    int num_rand_samples,
    float variance,
    float second_variance,
    float bounding_cube_dim,
    int num_votes) {
  float stdv = sqrt(variance);

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);
  std::vector<Eigen::Vector3f> xyz_used;
  std::vector<Eigen::Vector3f> second_samples;

  std::random_device rd;
  std::mt19937 rng(rd());
  // This line is of no use, while vertices should be the second group of sampled surface points.
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);
  std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

  // xyz_surf refers to the sampled points on surfaces.
  for (unsigned int i = 0; i < xyz_surf.size(); i++) {
    Eigen::Vector3f surface_p = xyz_surf[i];
      
    // samp1 and samp2 should be two variations (positive or negative about the surface) of xyz_surf[i].
    Eigen::Vector3f samp1 = surface_p;
    Eigen::Vector3f samp2 = surface_p;

    for (int j = 0; j < 3; j++) {
      samp1[j] += perterb_norm(rng);
      samp2[j] += perterb_second(rng);
    }

    xyz.push_back(samp1);
    xyz.push_back(samp2);
  }

  // bounding_cube_dim is set to 2, which rand_dist has a range of (0,1).
  // num_rand_samples refers to samples that randomly distribute in the domain. To be noted, points that exactly on the surface, in other words points whose sdf values equal to 0, are not included in the training set.
  // This indicates that the geometry is normalised to a unit sphere in the main function.
  for (int s = 0; s < (int)(num_rand_samples); s++) {
    xyz.push_back(Eigen::Vector3f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
  }

  // now compute sdf for each xyz sample
  // num_votes is set to 11. According to my experience, num_votes should refer to a hyperparameter in kdTree.
  // kdTree is included in nanoflann. In this case, it is a 3dTree, while we need a 2dTree.
  for (int s = 0; s < (int)xyz.size(); s++) {
    Eigen::Vector3f samp_vert = xyz[s];
    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);
      
    // cl_indices and cl_distances are set to be empty while filled during knnSearch.
    kdTree.knnSearch(samp_vert.data(), num_votes, cl_indices.data(), cl_distances.data());

    int num_pos = 0;
    float sdf;

    for (int ind = 0; ind < num_votes; ind++) {
      uint32_t cl_ind = cl_indices[ind];
      Eigen::Vector3f cl_vert = vertices[cl_ind];
      Eigen::Vector3f ray_vec = samp_vert - cl_vert;
      float ray_vec_leng = ray_vec.norm();

      if (ind == 0) {
        // if close to the surface, use point plane distance
        if (ray_vec_leng < stdv)
          sdf = fabs(normals[cl_ind].dot(ray_vec));
        else
          sdf = ray_vec_leng;
      }

      float d = normals[cl_ind].dot(ray_vec / ray_vec_leng);
      if (d > 0)
        num_pos++;
    }

    // all or nothing , else ignore the point. In our case, we don't care about this determine statement cuz our shapes are all watertight
    if ((num_pos == 0) || (num_pos == num_votes)) {
      xyz_used.push_back(samp_vert);
      if (num_pos <= (num_votes / 2)) {
        sdf = -sdf;
      }
      sdfs.push_back(sdf);
    }
  }

  xyz = xyz_used;
}

void writeSDFToNPY(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename) {
  unsigned int num_vert = xyz.size();
  std::vector<float> data(num_vert * 4);
  int data_i = 0;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    for (int j = 0; j < 3; j++)
      data[data_i++] = v[j];
    data[data_i++] = s;
  }

  cnpy::npy_save(filename, &data[0], {(long unsigned int)num_vert, 4}, "w");
}

void writeSDFToNPZ(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename,
    bool print_num = false) {
  unsigned int num_vert = xyz.size();
  std::vector<float> pos;
  std::vector<float> neg;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    if (s > 0) {
      for (int j = 0; j < 3; j++)
        pos.push_back(v[j]);
      pos.push_back(s);
    } else {
      for (int j = 0; j < 3; j++)
        neg.push_back(v[j]);
      neg.push_back(s);
    }
  }

  cnpy::npz_save(filename, "pos", &pos[0], {(long unsigned int)(pos.size() / 4.0), 4}, "w");
  cnpy::npz_save(filename, "neg", &neg[0], {(long unsigned int)(neg.size() / 4.0), 4}, "a");
  if (print_num) {
    std::cout << "pos num: " << pos.size() / 4.0 << std::endl;
    std::cout << "neg num: " << neg.size() / 4.0 << std::endl;
  }
}

void writeSDFToPLY(
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    std::string filename,
    bool neg_only = true,
    bool pos_only = false) {
  int num_verts;
  if (neg_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s <= 0)
        num_verts++;
    }
  } else if (pos_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s >= 0)
        num_verts++;
    }
  } else {
    num_verts = xyz.size();
  }

  std::ofstream plyFile;
  plyFile.open(filename);
  plyFile << "ply\n";
  plyFile << "format ascii 1.0\n";
  plyFile << "element vertex " << num_verts << "\n";
  plyFile << "property float x\n";
  plyFile << "property float y\n";
  plyFile << "property float z\n";
  plyFile << "property uchar red\n";
  plyFile << "property uchar green\n";
  plyFile << "property uchar blue\n";
  plyFile << "end_header\n";

  for (int i = 0; i < (int)sdfs.size(); i++) {
    Eigen::Vector3f v = xyz[i];
    float sdf = sdfs[i];
    bool neg = (sdf <= 0);
    bool pos = (sdf >= 0);
    if (neg)
      sdf = -sdf;
    int sdf_i = std::min((int)(sdf * 255), 255);
    if (!neg_only && pos)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << 0 << " " << 0 << " " << sdf_i << "\n";
    if (!pos_only && neg)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << sdf_i << " " << 0 << " " << 0 << "\n";
  }
  plyFile.close();
}

int main(int argc, char** argv) {
  std::string meshFileName;
  bool vis = false;

  std::string npyFileName;
  std::string plyFileNameOut;
  std::string spatial_samples_npz;
  bool save_ply = true;
  bool test_flag = false;
  float variance = 0.005;
  int num_sample = 500000;
  float rejection_criteria_obs = 0.02f;
  float rejection_criteria_tri = 0.03f;
  float num_samp_near_surf_ratio = 47.0f / 50.0f;

  // Similar to the Arg_Parser module in Python
  CLI::App app{"PreprocessMesh"};
  app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
  app.add_flag("-v", vis, "enable visualization");
  app.add_option("-o", npyFileName, "Save npy pc to here")->required();
  app.add_option("--ply", plyFileNameOut, "Save ply pc to here");
  app.add_option("-s", num_sample, "Save ply pc to here");
  app.add_option("--var", variance, "Set Variance");
  app.add_flag("--sply", save_ply, "save ply point cloud for visualization");
  app.add_flag("-t", test_flag, "test_flag");
  app.add_option("-n", spatial_samples_npz, "spatial samples from file");

  CLI11_PARSE(app, argc, argv);

  if (test_flag)
    variance = 0.05;

  float second_variance = variance / 10;
  std::cout << "variance: " << variance << " second: " << second_variance << std::endl;
  if (test_flag) {
    second_variance = variance / 100;
    num_samp_near_surf_ratio = 45.0f / 50.0f;
    num_sample = 250000;
  }

  std::cout << spatial_samples_npz << std::endl;

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

  // According to preprocess.py and the source code of pangolin::Geometry.cpp, LoadGeometry will load *.obj files.
  // According to Pangolin/geometry_obj.cpp, LoadGeometry in default loads face elements. Not sure whether we could easily process line elements. But we do have some tricks to take advantage of these developed codes.
  pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);

  std::cout << geom.objects.size() << " objects" << std::endl;

  // linearize the object indices
  {
      
    int total_num_faces = 0;

    for (const auto& object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {
        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        total_num_faces += ibo.h;
      }
    }

    //      const int total_num_indices = total_num_faces * 3;
    // ManagedImage refers to images that manage their own memory, storing a strong pointer to the memory.
      
    // Seemingly this part is just to improve the efficiency of memory usage. new_ibo, currently being an empty Image, is created to save vertex_indices.
    pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t), total_num_faces);

    pangolin::Image<uint32_t> new_ibo =
        new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);

    int index = 0;

    for (const auto& object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {
        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        for (int i = 0; i < ibo.h; ++i) {
          new_ibo.Row(index).CopyFrom(ibo.Row(i));
          ++index;
        }
      }
    }

    // According to the source code of .SubImage, the operation of SubImage(0,0,3,total_num_faces) takes the indices of three vertices of all triangles.
    // In our blank optimisation case, it should be SubImage(0,0,2,total_num_lines) since we use line elements.
    geom.objects.clear();
    auto faces = geom.objects.emplace(std::string("mesh"), pangolin::Geometry::Element());

    faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

    faces->second.CopyFrom(new_buffer);

    // Faces are linearized and the vertex_indices saved in new_ibo are converted to the attributes of faces.
    new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);
    faces->second.attributes["vertex_indices"] = new_ibo;
  }

  // remove textures (just the next line)
  geom.textures.clear();

  pangolin::Image<uint32_t> modelFaces = pangolin::get<pangolin::Image<uint32_t>>(
      geom.objects.begin()->second.attributes["vertex_indices"]);

  float max_dist = BoundingCubeNormalization(geom, true);

  //get vertices2
  int num_samp_near_surf = (int)(47 * num_sample / 50);
  std::cout << "num_samp_near_surf: " << num_samp_near_surf << std::endl;
  std::vector<Eigen::Vector3f> vertices2;
  SampleFromSurface(geom, vertices2, num_samp_near_surf / 2);

  KdVertexList kdVerts(vertices2);
  KdVertexListTree kdTree_surf(3, kdVerts);
  kdTree_surf.buildIndex();

  std::vector<Eigen::Vector3f> xyz;
  std::vector<Eigen::Vector3f> xyz_surf;
  std::vector<float> sdf;
  
  
  // xyz_surf refers to surfpts in the function of SampleFromSurface. This should be passed by address.
  // Another group of surface points should be sampled to calculate sdf values.
  SampleFromSurface(geom, xyz_surf, num_samp_near_surf / 2);

  // xyz, on the contrary, is created as an empty vector, which will be filled in the SampleSDFNearSurface function below.
  auto start = std::chrono::high_resolution_clock::now();
  SampleSDFNearSurface(
      kdTree_surf,
      vertices2,
      xyz_surf,
      normals2,
      xyz,
      sdf,
      num_sample - num_samp_near_surf,
      variance,
      second_variance,
      2,
      11);

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
  std::cout << elapsed << std::endl;

  if (save_ply) {
    writeSDFToPLY(xyz, sdf, plyFileNameOut, false, true);
  }

  std::cout << "num points sampled: " << xyz.size() << std::endl;
  std::size_t save_npz = npyFileName.find("npz");
  if (save_npz == std::string::npos)
    writeSDFToNPY(xyz, sdf, npyFileName);
  else {
    writeSDFToNPZ(xyz, sdf, npyFileName, true);
  }

  return 0;
}
