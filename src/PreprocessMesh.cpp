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


#include <nanoflann.hpp>
#include <CLI/CLI.hpp>
#include <cnpy.h>


extern pangolin::GlSlProgram GetShaderProgram();

struct KdVertexList {
public:

    KdVertexList(const std::vector<Eigen::Vector3f> & points)
            : points_(points) { }

    inline size_t kdtree_get_point_count() const {
        return points_.size();
    }

    inline float kdtree_distance(const float * p1, const size_t idx_p2, size_t /*size*/) const {
        Eigen::Map<const Eigen::Vector3f> p(p1);
        return (p - points_[idx_p2]).squaredNorm();
    }

    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return points_[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

private:

    std::vector<Eigen::Vector3f> points_;

};

using KdVertexListTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdVertexList>, KdVertexList, 3, int>;

std::vector<Eigen::Vector3f> EquiDistPointsOnSphere(int num_samples, float radius) {

  float pi = atan(1) * 4;
  float rnd = 1.;

  std::vector<Eigen::Vector3f> points(num_samples);
  float offset = 2. / num_samples;

  float increment = pi * (3 - sqrt(5));

  for (int i = 0; i < num_samples; i++) {
    float y = ((i * offset) - 1) + (offset / 2);
    float r = sqrt(1 - pow(y, 2));

    float phi = (i + rnd) * increment;

    float x = cos(phi) * r;
    float z = sin(phi) * r;

    x = x * radius;
    y = y * radius;
    z = z * radius;

    points[i] = Eigen::Vector3f(x, y, z);
  }

  return points;
}

std::vector<Eigen::Vector4f> ValidPointsFromIm(const pangolin::Image<Eigen::Vector4f>& verts) {
  std::vector<Eigen::Vector4f> points;
  Eigen::Vector4f v;

  for (unsigned int w = 0; w < verts.w; w++) {
    for (unsigned int h = 0; h < verts.h; h++) {
      v = verts(w, h);
      if (v[3] == 0.0f) {
        continue;
      }
      points.push_back(v);
    }
  }
  return points;
}

std::vector<Eigen::Vector4f> ValidPointsAndTrisFromIm(
    const pangolin::Image<Eigen::Vector4f>& pix_norms,
    std::vector<Eigen::Vector4f>& tris,
    int& total_obs,
    int& wrong_obs) {
  std::vector<Eigen::Vector4f> points;
  Eigen::Vector4f n;

  for (unsigned int w = 0; w < pix_norms.w; w++) {
    for (unsigned int h = 0; h < pix_norms.h; h++) {
      n = pix_norms(w, h);
      if (n[3] == 0.0f)
        continue;
      total_obs++;
      unsigned int tri_ind = (int)(n[3] + 0.01f) - 1;
      std::cout << tri_ind << std::endl;
      Eigen::Vector4f tri_track = tris[tri_ind];
      if (tri_track[3] == 0.0f)
        tris[tri_ind] = n;
      else if (tri_track[3] > 0.0f) {
        float dot = tri_track.head<3>().dot(n.head<3>());
        if (dot < 0.0f) {
          tris[tri_ind][3] = -1.0f;
          wrong_obs++;
        }
      } else if (tri_track[3] < 0.0f) {
        wrong_obs++;
        // n = Eigen::Vector4f(1.0, 0.0f,0.0f, 1.0f);
      }
      points.push_back(n);
    }
  }
  return points;
}

float TriangleArea(Eigen::Vector3f a, Eigen::Vector3f b, Eigen::Vector3f c) {
  Eigen::Vector3f ab = b - a;
  Eigen::Vector3f ac = c - a;

  float pi = 3.141592f;

  float costheta = ab.dot(ac) / (ab.norm() * ac.norm());

  if (costheta < -1) // meaning theta is pi
    costheta = cos(pi * 179 / 180);
  else if (costheta > 1) // meaning theta is zero
    costheta = cos(pi * 1 / 180);
  if (costheta < -1 || costheta > 1) {
    std::cout << costheta << std::endl;
    std::cout << ab << " " << ac << std::endl;
  }
  float sintheta = sqrt(1 - costheta * costheta);

  return 0.5 * ab.norm() * ac.norm() * sintheta;
}

void SampleFromTriangle(
    Eigen::Vector3f a,
    Eigen::Vector3f b,
    Eigen::Vector3f c,
    std::vector<Eigen::Vector3f>& surfpts,
    int num_sample) {
  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);

  for (int i = 0; i < num_sample; i++) {
    float r1 = rand_dist(generator);
    float r2 = rand_dist(generator);

    Eigen::Vector3f p = (1 - sqrt(r1)) * a + sqrt(r1) * (1 - r2) * b + r2 * sqrt(r1) * c;
    surfpts.push_back(p);
  }
}


void SampleFromSurface2(pangolin::Geometry& geom, std::vector<Eigen::Vector3f>& surfpts, int num_sample) {
  float total_area = 0.0f;

  std::vector<float> cdf_by_area;

  std::vector<Eigen::Vector3i> linearized_faces;

  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {

      pangolin::Image<uint32_t> ibo = pangolin::get<pangolin::Image<uint32_t> >(it_vert_indices->second);

      for (int i = 0; i < ibo.h; ++i) {
          linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
      }

    }
  }

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float> >(geom.buffers["geometry"].attributes["vertex"]);

  for (const Eigen::Vector3i & face : linearized_faces) {

    float area = TriangleArea(
      (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
      (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
      (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2)))
    );

    if (isnan(area)) {
      area = 0.f;
    }

    total_area += area;

    if (cdf_by_area.empty()) {

      cdf_by_area.push_back(area);

    } else {

      cdf_by_area.push_back(cdf_by_area.back() + area);

    }

  }

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_area);

  while ((int)surfpts.size() < num_sample) {
    float tri_sample = rand_dist(generator);
    std::vector<float>::iterator tri_index_iter =
        lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
    int tri_index = tri_index_iter - cdf_by_area.begin();

    const Eigen::Vector3i & face = linearized_faces[tri_index];

    SampleFromTriangle(
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))),
        surfpts,
        1);
  }
}

float BoundingCubeNormalization(
    pangolin::Geometry& geom,
    bool fit_to_unit_sphere,
    float max_distance_buffer = 1.03f) {
  float x_min = 1000000, x_max = -1000000, y_min = 1000000, y_max = -1000000, z_min = 1000000,
        z_max = -1000000;

  pangolin::Image<float> vertices =
      pangolin::get<pangolin::Image<float> >(geom.buffers["geometry"].attributes["vertex"]);

  const std::size_t num_vertices = vertices.h;

  ///////// Only consider vertices that were used in some face
  std::vector<unsigned char> vertex_used(num_vertices, 0);
  // turn to true if the vertex is used
  for (const auto& object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {

      pangolin::Image<uint32_t> ibo = pangolin::get<pangolin::Image<uint32_t> >(it_vert_indices->second);

      for (int i = 0; i < ibo.h; ++i) {
        for (int j = 0; j < 3; ++j) {
          vertex_used[ibo(j, i)] = 1;
        }
      }

    }
  }
 /////////

  // compute min max in each dimension
  for (size_t i = 0; i < num_vertices; i++) {
    // pass when it's not used.
    if (vertex_used[i] == 0)
      continue;
    x_min = fmin(x_min, vertices(0, i));
    y_min = fmin(y_min, vertices(1, i));
    z_min = fmin(z_min, vertices(2, i));
    x_max = fmax(x_max, vertices(0, i));
    y_max = fmax(y_max, vertices(1, i));
    z_max = fmax(z_max, vertices(2, i));
  }

  const float x_center = (x_max + x_min) / 2.0f;
  const float y_center = (y_max + y_min) / 2.0f;
  const float z_center = (z_max + z_min) / 2.0f;

  std::cout << "center: " << x_center << " " << y_center << " " << z_center << std::endl;

  // make the mean zero
  float max_distance = -1.0f;
  for (size_t i = 0; i < num_vertices; i++) {
    // pass when it's not used.
    if (vertex_used[i] == false)
      continue;
    vertices(0, i) -= x_center;
    vertices(1, i) -= y_center;
    vertices(2, i) -= z_center;

    const float dist = Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(i)).norm();
    max_distance = std::max(max_distance, dist);
  }

  // add some buffer
  max_distance *= max_distance_buffer;

  std::cout << "max distance: " << max_distance << std::endl;

  if (fit_to_unit_sphere) {
    for (size_t i = 0; i < num_vertices; i++) {
      vertices(0, i) /= max_distance;
      vertices(1, i) /= max_distance;
      vertices(2, i) /= max_distance;
    }
    max_distance = 1;
  }

  return max_distance;
}

void SampleSDFNearSurface(
    KdVertexListTree& kdTree,
    std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    int num_samples,
    float variance,
    float bounding_cube_dim,
    int num_votes,
    float rand_prop_mult,
    bool uniform_sample,
    bool add_surface_points = true) {
  float stdv = sqrt(variance);

  float rand_samp_prop = 1 / rand_prop_mult;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);

  for (int s = 0; s < (int)(num_samples * rand_samp_prop); s++) {
    xyz.push_back(Eigen::Vector3f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
  }

  std::vector<Eigen::Vector3f> xyz_used;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);

  for (int s = 0; s < num_samples - (int)(num_samples * rand_samp_prop); s++) {
    int samp_ind = vert_ind(rng);
    Eigen::Vector3f samp_vert = vertices[samp_ind];

    if (!uniform_sample) {
      // Perterb the sampled vertex
      for (int i = 0; i < 3; i++) {
        samp_vert[i] += perterb_norm(rng);
      }
    } else {
      Eigen::Vector3f direction(rand_dist(generator), rand_dist(generator), rand_dist(generator));
      direction = direction / direction.norm();
      float magnitude = rand_dist(generator) * variance;
      samp_vert += magnitude * direction;
    }
    xyz.push_back(samp_vert);
  }

  // now compute sdf for each xyz sample
  for (int s = 0; s < num_samples; s++) {
    Eigen::Vector3f samp_vert = xyz[s];

    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);
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
        // sdf = ray_vec_leng;
        if (add_surface_points) {
          // add the closest surface points
          xyz_used.push_back(cl_vert);
          sdfs.push_back(0);
        }
      }

      float d = normals[cl_ind].dot(ray_vec / ray_vec_leng);
      if (d > 0)
        num_pos++;
    }

    // all or nothing , else ignore the point
    if ((num_pos == 0) || (num_pos == num_votes)) {
      xyz_used.push_back(samp_vert);
      if (num_pos <= (num_votes / 2)) {
        sdf = -sdf;
      }
      sdfs.push_back(sdf);
    }

    //        if (num_pos <= int(num_votes / 2))
    //            sdf = -sdf;
    //        xyz_used.push_back(samp_vert);
    //        sdfs.push_back(sdf);
  }

  xyz = xyz_used;
}

void SampleSDFNearSurface2(
    KdVertexListTree& kdTree_surf,
    KdVertexListTree& kdTree_all,
    std::vector<Eigen::Vector3f>& vertices,
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& xyz,
    std::vector<float>& sdfs,
    int num_samples,
    float variance,
    float bounding_cube_dim,
    int num_votes,
    float rand_prop_mult,
    bool uniform_sample,
    int index_border,
    bool add_surface_points = true,
    bool select_min_angle = true) {
  std::cout << "index_border " << index_border << std::endl;
  float stdv = sqrt(variance);

  float rand_samp_prop = 1 / rand_prop_mult;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);

  for (int s = 0; s < (int)(num_samples * rand_samp_prop); s++) {
    xyz.push_back(Eigen::Vector3f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
  }

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);

  std::vector<Eigen::Vector3f> surface_samples;

  for (int s = 0; s < num_samples - (int)(num_samples * rand_samp_prop); s++) {
    int samp_ind = vert_ind(rng);
    Eigen::Vector3f samp_vert = vertices[samp_ind];

    if (add_surface_points)
      surface_samples.push_back(samp_vert);

    if (!uniform_sample) {
      // Perterb the sampled vertex
      for (int i = 0; i < 3; i++) {
        samp_vert[i] += perterb_norm(rng);
      }
    } else {
      Eigen::Vector3f direction(rand_dist(generator), rand_dist(generator), rand_dist(generator));
      direction = direction / direction.norm();
      float magnitude = rand_dist(generator) * variance;
      samp_vert += magnitude * direction;
    }
    xyz.push_back(samp_vert);
  }

  // now compute sdf for each xyz sample
  for (int s = 0; s < num_samples; s++) {
    Eigen::Vector3f samp_vert = xyz[s];
    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);
    kdTree_all.knnSearch(samp_vert.data(), num_votes, cl_indices.data(), cl_distances.data());
    int closest_surf;
    float closest_surf_dist;
    kdTree_surf.knnSearch(samp_vert.data(), 1, &closest_surf, &closest_surf_dist);
    Eigen::Vector3f closest_vert = vertices[closest_surf];
    Eigen::Vector3f closest_ray_vec = samp_vert - closest_vert;

    int num_pos = 0;
    float sdf = closest_ray_vec.norm();
    if (sdf < stdv)
      sdf = fabs(normals[closest_surf].dot(closest_ray_vec));

    for (int ind = 0; ind < num_votes; ind++) {
      uint32_t cl_ind = cl_indices[ind];

      if ((int)cl_ind >= index_border)
        num_pos++;
    }

    if (num_pos <= (int)(num_votes * 2 / 3))
      sdf = -sdf;

    sdfs.push_back(sdf);
  }

  if (add_surface_points) {
    for (unsigned int s = 0; s < surface_samples.size(); s++) {
      xyz.push_back(surface_samples[s]);
      sdfs.push_back(0);
    }
  }
}

void SampleSDFNearSurface5(
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
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);
  std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

  for (unsigned int i = 0; i < xyz_surf.size(); i++) {
    Eigen::Vector3f surface_p = xyz_surf[i];
    Eigen::Vector3f samp1 = surface_p;
    Eigen::Vector3f samp2 = surface_p;

    for (int j = 0; j < 3; j++) {
      samp1[j] += perterb_norm(rng);
      samp2[j] += perterb_second(rng);
    }

    xyz.push_back(samp1);
    xyz.push_back(samp2);
  }

  for (int s = 0; s < (int)(num_rand_samples); s++) {
    xyz.push_back(Eigen::Vector3f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
  }

  // now compute sdf for each xyz sample
  for (int s = 0; s < (int)xyz.size(); s++) {
    Eigen::Vector3f samp_vert = xyz[s];
    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);
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

    // all or nothing , else ignore the point
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

void writeSDFToNPY(std::vector<Eigen::Vector3f>& xyz, std::vector<float>& sdfs, std::string filename) {
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

void writeSDFToNPZ(std::vector<Eigen::Vector3f>& xyz, std::vector<float>& sdfs, std::string filename) {
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
  float variance = 0.005 / 2;
  int num_sample = 1000;
  float rejection_criteria_obs = 0.02f;
  float rejection_criteria_tri = 0.03f;

  CLI::App app{"SnowSculptor"};
  app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
  app.add_flag("-v", vis, "enable visualization");
  app.add_option("-o", npyFileName, "Save npy pc to here")->required();
  app.add_option("--ply", plyFileNameOut, "Save ply pc to here");
  app.add_option("-s", num_sample, "Save ply pc to here");
  app.add_option("--var", variance, "Save ply pc to here");
  app.add_flag("--sply", save_ply, "save ply point cloud for visualization");
  app.add_flag("-t", test_flag, "test_flag");
  app.add_option("-n", spatial_samples_npz, "spatial samples from file");

  CLI11_PARSE(app, argc, argv);

  if (test_flag) {
    // figure out this
    variance = 0.01;
  }
  float second_variance = variance / 10;
  if (test_flag)
    second_variance = variance / 20;

  std::cout << spatial_samples_npz << std::endl;

  pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);

  std::cout << geom.objects.size() << " objects" << std::endl;

  // linearize the object indices
  {
      int total_num_faces = 0;

      for (const auto& object : geom.objects) {
        auto it_vert_indices = object.second.attributes.find("vertex_indices");
        if (it_vert_indices != object.second.attributes.end()) {

          pangolin::Image<uint32_t> ibo = pangolin::get<pangolin::Image<uint32_t> >(it_vert_indices->second);

          total_num_faces += ibo.h;

        }
      }

//      const int total_num_indices = total_num_faces * 3;
      pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t), total_num_faces);

      pangolin::Image<uint32_t> new_ibo = new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0,0,3,total_num_faces);

      int index = 0;

      for (const auto& object : geom.objects) {
        auto it_vert_indices = object.second.attributes.find("vertex_indices");
        if (it_vert_indices != object.second.attributes.end()) {

          pangolin::Image<uint32_t> ibo = pangolin::get<pangolin::Image<uint32_t> >(it_vert_indices->second);

          for (int i = 0; i < ibo.h; ++i) {

            new_ibo.Row(index).CopyFrom(ibo.Row(i));
            ++index;

          }

        }
      }

      geom.objects.clear();
      auto faces = geom.objects.emplace(std::string("mesh"), pangolin::Geometry::Element());

      faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

      faces->second.CopyFrom(new_buffer);

      new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(0,0,3,total_num_faces);
      faces->second.attributes["vertex_indices"] = new_ibo;

  }

  pangolin::Image<uint32_t> modelFaces =
      pangolin::get<pangolin::Image<uint32_t> >(geom.objects.begin()->second.attributes["vertex_indices"]);

  float max_dist = BoundingCubeNormalization(geom, true);

  if (vis)
    pangolin::CreateWindowAndBind("Main", 640, 480);
  else
    pangolin::CreateWindowAndBind("Main", 1, 1);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_DITHER);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POLYGON_SMOOTH);
  glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
  glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
  glDisable(GL_MULTISAMPLE_ARB);
  glShadeModel(GL_FLAT);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      //                pangolin::ProjectionMatrix(640,480,420,420,320,240,0.05,100),
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, -max_dist, max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));
  pangolin::OpenGlRenderState s_cam2(
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, max_dist, -max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);

  pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

//  pangolin::GlSlProgram prog;
//  prog.AddShaderFromFile(
//      pangolin::GlSlVertexShader, "/tmp/vertex-normal.vert");
//  prog.AddShaderFromFile(
//      pangolin::GlSlFragmentShader, "/tmp/vertex-normal.frag");
//  prog.AddShaderFromFile(
//      pangolin::GlSlGeometryShader, "/tmp/vertex-normal.geom");
//  prog.Link();

  pangolin::GlSlProgram prog = GetShaderProgram();

  if (vis) {
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      //        glEnable(GL_CULL_FACE);
      //        glCullFace(GL_FRONT);

      d_cam.Activate(s_cam);

      prog.Bind();
      prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
      prog.SetUniform("V", s_cam.GetModelViewMatrix());

      pangolin::GlDraw(prog, gl_geom, nullptr);
      prog.Unbind();

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }

  // Create Framebuffer with attached textures
  size_t w = 400;
  size_t h = 400;
  pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
  pangolin::GlTexture normals(w, h, GL_RGBA32F);
  pangolin::GlTexture vertices(w, h, GL_RGBA32F);
  pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

  // View points around a sphere.
  std::vector<Eigen::Vector3f> views = EquiDistPointsOnSphere(100, max_dist * 1.1);

  std::vector<Eigen::Vector4f> point_normals;
  std::vector<Eigen::Vector4f> point_verts;

  size_t num_tri = modelFaces.h;
  std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
  for (size_t j = 0; j < num_tri; j++)
    tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
  int total_obs = 0;
  int wrong_obs = 0;

  for (unsigned int v = 0; v < views.size(); v++) {
    // change camera location
    s_cam2.SetModelViewMatrix(
        pangolin::ModelViewLookAt(views[v][0], views[v][1], views[v][2], 0, 0, 0, pangolin::AxisY));
    // Draw the scene to the framebuffer
    framebuffer.Bind();
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prog.Bind();
    prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
    prog.SetUniform("V", s_cam2.GetModelViewMatrix());
    prog.SetUniform("ToWorld", s_cam2.GetModelViewMatrix().Inverse());
    prog.SetUniform("slant_thr", -1.0f, 1.0f);
    prog.SetUniform("ttt", 1.0, 0, 0, 1);
    pangolin::GlDraw(prog, gl_geom, nullptr);
    prog.Unbind();

    framebuffer.Unbind();

    pangolin::TypedImage img_normals;
    normals.Download(img_normals);
    std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
        img_normals.UnsafeReinterpret<Eigen::Vector4f>(),
        tri_id_normal_test,
        total_obs,
        wrong_obs);
    point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());

    pangolin::TypedImage img_verts;
    vertices.Download(img_verts);
    std::vector<Eigen::Vector4f> im_verts =
        ValidPointsFromIm(img_verts.UnsafeReinterpret<Eigen::Vector4f>());
    point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());

  }

  int bad_tri = 0;
  for (unsigned int t; t < tri_id_normal_test.size(); t++) {
    if (tri_id_normal_test[t][3] < 0.0f)
      bad_tri++;
  }

  std::cout << meshFileName << std::endl;
  std::cout << (float)(wrong_obs) / float(total_obs) << std::endl;
  std::cout << (float)(bad_tri) / float(num_tri) << std::endl;

  float wrong_ratio = (float)(wrong_obs) / float(total_obs);
  float bad_tri_ratio = (float)(bad_tri) / float(num_tri);

  if (wrong_ratio > rejection_criteria_obs || bad_tri_ratio > rejection_criteria_tri) {
    std::cout << "mesh rejected" << std::endl;
    return 0;
  }

  std::vector<Eigen::Vector3f> vertices2;
  //    std::vector<Eigen::Vector3f> vertices_all;
  std::vector<Eigen::Vector3f> normals2;

  for (unsigned int v = 0; v < point_verts.size(); v++) {
    vertices2.push_back(point_verts[v].head<3>());
    normals2.push_back(point_normals[v].head<3>());
  }

  KdVertexList kdVerts(vertices2);
  KdVertexListTree kdTree_surf(3, kdVerts);
  kdTree_surf.buildIndex();

  std::vector<Eigen::Vector3f> xyz;
  std::vector<Eigen::Vector3f> xyz_surf;
  std::vector<float> sdf;
  int num_samp_near_surf = (int)(3 * num_sample / 50);
  SampleFromSurface2(geom, xyz_surf, num_samp_near_surf / 2);


  auto start = std::chrono::high_resolution_clock::now();
  SampleSDFNearSurface5(
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

    writeSDFToPLY(xyz, sdf, plyFileNameOut, true, false);

  }

  std::cout << "num points sampled: " << xyz.size() << std::endl;
  std::size_t save_npz = npyFileName.find("npz");
  if (save_npz == std::string::npos)
    writeSDFToNPY(xyz, sdf, npyFileName);
  else {
    writeSDFToNPZ(xyz, sdf, npyFileName);
  }

  return 0;
}
