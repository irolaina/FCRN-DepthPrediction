/*
  I/O interface class for loading, storing and manipulating
  depth maps in KITTI format. This file requires libpng
  and libpng++ to be installed for accessing png files. More
  detailed format specifications can be found in the readme.txt

  (c) Andreas Geiger
*/

#ifndef IO_DEPTH_H
#define IO_DEPTH_H

#include <string.h>
#include <stdint.h>
#include <png++/png.hpp>
#include "log_colormap.h"

class DepthImage {

public:

  // default constructor
  DepthImage () {
    data_   = 0;
    width_  = 0;
    height_ = 0;
  }

  // construct depth image from png file
  DepthImage (const std::string file_name) {
    readDepthMap(file_name);
  }

  // copy constructor
  DepthImage (const DepthImage &D) {
    width_  = D.width_;
    height_ = D.height_;
    data_   = (float*)malloc(width_*height_*sizeof(float));
    memcpy(data_,D.data_,width_*height_*sizeof(float));
  }

  // construct depth image from data
  DepthImage (const float* data, const int32_t width, const int32_t height) : width_(width), height_(height) {
    data_ = (float*)malloc(width*height*sizeof(float));
    memcpy(data_,data,width*height*sizeof(float));
  }

  // construct empty (= all pixels invalid) depth map of given width / height
  DepthImage (const int32_t width, const int32_t height) : width_(width), height_(height) {
    data_ = (float*)malloc(width*height*sizeof(float));
    for (int32_t i=0; i<width*height; i++)
      data_[i] = -1;
  }

  // deconstructor
  virtual ~DepthImage () {
    if (data_) { free(data_); data_ = 0; }
  }

  // assignment operator, copies contents of D
  DepthImage& operator= (const DepthImage &D) {
    if (this!=&D) {
      if (D.width_!=width_ || D.height_!=height_) {
        free(data_);
        width_  = D.width_;
        height_ = D.height_;
        data_   = (float*)malloc(width_*height_*sizeof(float));
      }
      memcpy(data_,D.data_,width_*height_*sizeof(float));
    }
    return *this;
  }

  // read depth image from png file
  void read (const std::string file_name) {
    if (data_) { free(data_); data_ = 0; }
    readDepthMap(file_name);
  }

  // write depth image to png file
  void write (const std::string file_name) {
    writeDepthMap(file_name);
  }

  // write depth image to (grayscale printable) false color map
  // if max_depth<1, the scaling is determined from the maximum depth
  void writeColor (const std::string file_name,float max_depth=-1.0f) {
    if (max_depth<=1.0f)
      max_depth = std::max(maxDepth(),1.0f);
    writeFalseColors(file_name,max_depth);
  }

  // get depth at given pixel
  inline float getDepth (const int32_t u,const int32_t v) {
    return data_[v*width_+u];
  }

  // is depth valid
  inline bool isValid (const int32_t u,const int32_t v) {
    return data_[v*width_+u]>=0;
  }

  // set depth at given pixel
  inline void setDepth (const int32_t u,const int32_t v,const float val) {
    data_[v*width_+u] = val;
  }

  // is depth at given pixel to invalid
  inline bool setInvalid (const int32_t u,const int32_t v) {
    data_[v*width_+u] = -1;
  }

  // get maximal depth
  float maxDepth () {
    float max_depth = -1;
    for (int32_t i=0; i<width_*height_; i++)
      if (data_[i]>max_depth)
        max_depth = data_[i];
    return max_depth;
  }

  // simple arithmetic operations
  DepthImage operator+ (const DepthImage &B) {
    DepthImage C(*this);
    for (int32_t i=0; i<width_*height_; i++)
      C.data_[i] += B.data_[i];
    return C;
  }
  DepthImage operator- (const DepthImage &B) {
    DepthImage C(*this);
    for (int32_t i=0; i<width_*height_; i++)
      C.data_[i] -= B.data_[i];
    return C;
  }
  DepthImage abs () {
    DepthImage C(*this);
    for (int32_t i=0; i<width_*height_; i++)
      C.data_[i] = fabs(C.data_[i]);
    return C;
  }

  // interpolate all missing (=invalid) depths
  void interpolateBackground () {

    // for each row do
    for (int32_t v=0; v<height_; v++) {

      // init counter
      int32_t count = 0;

      // for each pixel do
      for (int32_t u=0; u<width_; u++) {

        // if depth valid
        if (isValid(u,v)) {

          // at least one pixel requires interpolation
          if (count>=1) {

            // first and last value for interpolation
            int32_t u1 = u-count;
            int32_t u2 = u-1;

            // set pixel to min depth
            if (u1>0 && u2<width_-1) {
              float d_ipol = std::min(getDepth(u1-1,v),getDepth(u2+1,v));
              for (int32_t u_curr=u1; u_curr<=u2; u_curr++)
                setDepth(u_curr,v,d_ipol);
            }
          }

          // reset counter
          count = 0;

        // otherwise increment counter
        } else {
          count++;
        }
      }

      // extrapolate to the left
      for (int32_t u=0; u<width_; u++) {
        if (isValid(u,v)) {
          for (int32_t u2=0; u2<u; u2++)
            setDepth(u2,v,getDepth(u,v));
          break;
        }
      }

      // extrapolate to the right
      for (int32_t u=width_-1; u>=0; u--) {
        if (isValid(u,v)) {
          for (int32_t u2=u+1; u2<=width_-1; u2++)
            setDepth(u2,v,getDepth(u,v));
          break;
        }
      }
    }

    // for each column do
    for (int32_t u=0; u<width_; u++) {

      // extrapolate to the top
      for (int32_t v=0; v<height_; v++) {
        if (isValid(u,v)) {
          for (int32_t v2=0; v2<v; v2++)
            setDepth(u,v2,getDepth(u,v));
          break;
        }
      }

      // extrapolate to the bottom
      for (int32_t v=height_-1; v>=0; v--) {
        if (isValid(u,v)) {
          for (int32_t v2=v+1; v2<=height_-1; v2++)
            setDepth(u,v2,getDepth(u,v));
          break;
        }
      }
    }
  }

  // compute error map of current image, given the
  // ground truth depth maps. stores result as color png image.
  png::image<png::rgb_pixel> errorImage (DepthImage &D_gt,bool log_colors=false) {
    png::image<png::rgb_pixel> image(width(),height());
    for (int32_t v=1; v<height()-1; v++) {
      for (int32_t u=1; u<width()-1; u++) {
        if (D_gt.isValid(u,v)) {
          png::rgb_pixel val;
          if (log_colors) {
            float d_err = fabs(getDepth(u,v)-D_gt.getDepth(u,v));
            float d_mag = fabs(D_gt.getDepth(u,v));
            //why this heuristic here??
            float n_err = std::min(d_err/3.0,20.0*d_err/d_mag);
            for (int32_t i=0; i<10; i++) {
              if (n_err>=LC[i][0] && n_err<LC[i][1]) {
                val.red   = (uint8_t)LC[i][2];
                val.green = (uint8_t)LC[i][3];
                val.blue  = (uint8_t)LC[i][4];
              }
            }
          } else {
            float d_err = std::min(fabs(getDepth(u,v)-D_gt.getDepth(u,v)),5.0)/5.0;
            val.red   = (uint8_t)(d_err*255.0);
            val.green = (uint8_t)(d_err*255.0);
            val.blue  = (uint8_t)(d_err*255.0);
            if (!D_gt.isValid(u,v)) {
              val.green = 0;
              val.blue  = 0;
            }
          }
          for (int32_t v2=v-1; v2<=v+1; v2++)
            for (int32_t u2=u-1; u2<=u+1; u2++)
              image.set_pixel(u2,v2,val);
        }
      }
    }
    return image;
  }

  // direct access to private variables
  float*  data   () { return data_; }
  int32_t width  () { return width_; }
  int32_t height () { return height_; }

private:

  void readDepthMap (const std::string file_name) {
    png::image<png::gray_pixel_16> image(file_name);
    width_  = image.get_width();
    height_ = image.get_height();
    data_   = (float*)malloc(width_*height_*sizeof(float));
    for (int32_t v=0; v<height_; v++) {
      for (int32_t u=0; u<width_; u++) {
        uint16_t val = image.get_pixel(u,v);
        if (val==0) setInvalid(u,v);
        else        setDepth(u,v,((float)val)/256.0);
      }
    }
  }

  void writeDepthMap (const std::string file_name) {
    png::image< png::gray_pixel_16 > image(width_,height_);
    for (int32_t v=0; v<height_; v++) {
      for (int32_t u=0; u<width_; u++) {
        if (isValid(u,v)) image.set_pixel(u,v,(uint16_t)(std::max(getDepth(u,v)*256.0,1.0)));
        else              image.set_pixel(u,v,0);
      }
    }
    image.write(file_name);
  }

  void writeFalseColors (const std::string file_name, float max_val) {

    // color map
    float map[8][4] = {{0,0,0,114},{0,0,1,185},{1,0,0,114},{1,0,1,174},
                       {0,1,0,114},{0,1,1,185},{1,1,0,114},{1,1,1,0}};
    float sum = 0;
    for (int32_t i=0; i<8; i++)
      sum += map[i][3];

    float weights[8]; // relative weights
    float cumsum[8];  // cumulative weights
    cumsum[0] = 0;
    for (int32_t i=0; i<7; i++) {
      weights[i]  = sum/map[i][3];
      cumsum[i+1] = cumsum[i] + map[i][3]/sum;
    }

    // create color png image
    png::image< png::rgb_pixel > image(width_,height_);

    // for all pixels do
    for (int32_t v=0; v<height_; v++) {
      for (int32_t u=0; u<width_; u++) {

        // get normalized value
        float val = std::min(std::max(getDepth(u,v)/max_val,0.0f),1.0f);

        // find bin
        int32_t i;
        for (i=0; i<7; i++)
          if (val<cumsum[i+1])
            break;

        // compute red/green/blue values
        float   w = 1.0-(val-cumsum[i])*weights[i];
        uint8_t r = (uint8_t)((w*map[i][0]+(1.0-w)*map[i+1][0]) * 255.0);
        uint8_t g = (uint8_t)((w*map[i][1]+(1.0-w)*map[i+1][1]) * 255.0);
        uint8_t b = (uint8_t)((w*map[i][2]+(1.0-w)*map[i+1][2]) * 255.0);

        // set pixel
        image.set_pixel(u,v,png::rgb_pixel(r,g,b));
      }
    }

    // write to file
    image.write(file_name);
  }

public:

  float  *data_;
  int32_t width_;
  int32_t height_;

};

#endif // IO_DEPTH_H

