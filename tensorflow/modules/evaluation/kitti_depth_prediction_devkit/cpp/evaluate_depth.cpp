#include <iostream>
#include <stdio.h>
#include <math.h>

#include "io_depth.h"
#include "utils.h"
//iterate over files in directory
#include "dirent.h"
#include <string.h>

using namespace std;

/** \brief Method to calculate depth average error.
  * \param D_gt the ground truth depth image
  * \param D_ipol the interpolated depth image to be benchmarked
  * \return mae between original and ground truth depth (2 entries: occluded and non-occluded)
  *
  */
std::vector<float> depthError (DepthImage &D_gt, DepthImage &D_ipol) {

  // check file size
  if (D_gt.width() != D_ipol.width() || D_gt.height() != D_ipol.height()) {
    cout << "ERROR: Wrong file size!" << endl;
    throw 1;
  }

  // extract width and height
  uint32_t width  = D_gt.width();
  uint32_t height = D_gt.height();

  //init errors
  // 1. mae
  // 2. rmse
  // 3. inverse mae
  // 4. inverse rmse
  // 5. log mae
  // 6. log rmse
  // 7. scale invariant log
  // 8. abs relative
  // 9. squared relative

  std::vector<float> errors(9, 0.f);
  //
  uint32_t num_pixels = 0;
  uint32_t num_pixels_result = 0;

  //log sum for scale invariant metric
  float logSum = 0.0;

  // for all pixels do
  for (uint32_t u = 0; u < width; u++) {
    for (uint32_t v = 0; v < height; v++) {
      if (D_gt.isValid(u, v)) {
        const float depth_ipol_m = D_ipol.getDepth(u, v);
        const float depth_gt_m   = D_gt.getDepth(u, v);

        //error if gt is valid
        const float d_err = fabs(depth_gt_m - depth_ipol_m);

        const float d_err_squared = d_err * d_err;
        const float d_err_inv = fabs( 1.0 / depth_gt_m - 1.0 / depth_ipol_m);
        const float d_err_inv_squared = d_err_inv * d_err_inv;
        const float d_err_log = fabs(log(depth_gt_m) - log(depth_ipol_m));
        const float d_err_log_squared = d_err_log * d_err_log;

        //mae
        errors[0] += d_err;
        //rmse
        errors[1] += d_err_squared;
        //inv_mae
        errors[2] += d_err_inv;
        //inv_rmse
        errors[3] += d_err_inv_squared;
        //log
        errors[4] += d_err_log;
        //log rmse
        errors[5] += d_err_log_squared;
        //log diff for scale invariant metric
        logSum += (log(depth_gt_m) - log(depth_ipol_m));
        //abs relative
        errors[7] += d_err/depth_gt_m;
        //squared relative
        errors[8] += d_err_squared/(depth_gt_m*depth_gt_m);

        //increase valid gt pixels
        num_pixels++;
      }
    } //end for v
  } //end for u

  // check number of pixels
  if (num_pixels == 0) {
    cout << "ERROR: Ground truth defect => Please write me an email!" << endl;
    throw 1;
  }

  //normalize mae
  errors[0] /= (float)num_pixels;
  //normalize and take sqrt for rmse
  errors[1] /= (float)num_pixels;
  errors[1] = sqrt(errors[1]);
  //normalize inverse absoulte error
  errors[2] /= (float)num_pixels;
  //normalize and take sqrt for inverse rmse
  errors[3] /= (float)num_pixels;
  errors[3] = sqrt(errors[3]);
  //normalize log mae
  errors[4] /= (float)num_pixels;
  //first normalize log rmse -> we need this result later
  const float normalizedSquaredLog = errors[5] / (float)num_pixels;
  errors[5] = sqrt(normalizedSquaredLog);
  //calculate scale invariant metric
  errors[6] = sqrt(normalizedSquaredLog - (logSum*logSum / ((float)num_pixels*(float)num_pixels)));
  //normalize abs relative
  errors[7] /= (float)num_pixels;
  //normalize squared relative
  errors[8] /= (float)num_pixels;
  // return errors
  return errors;
}

/** \brief Helper function for png file selection.
  * \param entry direct struct to be compared
  *
  */
int png_select(const dirent *entry)
{
  const char* fileName = entry->d_name;

  //check that this is not a directory
  if ((strcmp(fileName, ".")== 0) || (strcmp(fileName, "..") == 0))
    return false;

  /* Check for png filename extensions */
  const char* ptr = strrchr(fileName, '.');
  if ((ptr != NULL) && (strcmp(ptr, ".png") == 0))
    return true;
  else
    return false;
}

/** \brief Method to evaluate depth maps.
  * \param prediction_dir The directory containing predicted depth maps.
  * \return success If true, writes a txt file containing all depth error metrics.
  */
bool eval (string gt_img_dir, string prediction_dir) {
  // make sure all directories have ending slashes
  gt_img_dir += "/";
  prediction_dir += "/";

  // for all evaluation files do
  struct dirent **namelist_gt;
  struct dirent **namelist_prediction;
  int num_files_gt = scandir(gt_img_dir.c_str(), &namelist_gt, png_select, alphasort);
  int num_files_prediction = scandir(prediction_dir.c_str(), &namelist_prediction, png_select, alphasort);

  if( num_files_gt != num_files_prediction ){
    std::cout << "Number of groundtruth (" << num_files_gt << ") and prediction files (" << num_files_prediction << ") mismatch!" << std::endl;
    free(namelist_gt);
    free(namelist_prediction);
    return false;
  }
    std::cout << "Found " << num_files_gt << " groundtruth and " << num_files_prediction << " prediction files!" << std::endl;

  if( num_files_prediction < 0 ){
    perror("scandir");
  }

  // std::vector for storing the errors
  std::vector< std::vector<float> > errors_out;
  // create output directories
  system(("mkdir " + prediction_dir + "/errors_out/").c_str());
  system(("mkdir " + prediction_dir + "/errors_img/").c_str());
  system(("mkdir " + prediction_dir + "/depth_orig/").c_str());
  system(("mkdir " + prediction_dir + "/depth_ipol/").c_str());
  system(("mkdir " + prediction_dir + "/image_0/").c_str());

  for( int i = 0; i < num_files_prediction; ++i ){
    //Be aware: we use the same index here, the files have to be correctly sorted!!!!
    std::string fileName_gt = gt_img_dir + namelist_gt[i]->d_name;
    std::string fileName_prediction = prediction_dir + namelist_prediction[i]->d_name;

    if( strcmp(fileName_gt.c_str(), ".") == 0 || strcmp(fileName_gt.c_str(), "..") == 0 ) continue;
    std::string filePath = gt_img_dir + fileName_gt;
    //std::string fileName_gt = path.back();

    size_t lastindex = std::string(namelist_gt[i]->d_name).find_last_of(".");
    std::string prefix = std::string(namelist_gt[i]->d_name).substr(0, lastindex);

    // output
    std::cout << "Processing: " << prefix.c_str() << std::endl;

    // catch errors, when loading fails
    try {
      // load ground truth depth maps
      DepthImage D_gt(fileName_gt);

      // check file format
      if (!imageFormat(fileName_gt, png::color_type_gray, 16, D_gt.width(), D_gt.height())) {
        std::cout << "ERROR: Input must be png, 1 channel, 16 bit, " << D_gt.width() << " x " << D_gt.height() << "px" << std::endl;
        free(namelist_gt);
        free(namelist_prediction);
        return false;
      }
      // load prediction and interpolate missing values
      DepthImage D_orig(fileName_prediction);
      DepthImage D_ipol(D_orig);
      D_ipol.interpolateBackground();

      // add depth errors
      std::vector<float> errors_out_curr = depthError(D_gt, D_ipol);
      errors_out.push_back(errors_out_curr);

      // save detailed infos for first 20 images
      if (i < 20) {
        // save errors of error images to text file
        FILE *errors_out_file = fopen((prediction_dir + "/errors_out/" + prefix + ".txt").c_str(), "w");
        if (errors_out_file == NULL) {
          std::cout << "ERROR: Couldn't generate/store output statistics!" << std::endl;
          return false;
        }
        for (int32_t j = 0; j < errors_out_curr.size(); j++) {
          fprintf(errors_out_file, "%f ", errors_out_curr[j]);
        }
        fclose(errors_out_file);

        // save error image
        png::image<png::rgb_pixel> D_err = D_ipol.errorImage(D_gt, true);
        D_err.write(prediction_dir + "/errors_img/" + prefix + ".png");

        // compute maximum depth
        float max_depth = D_gt.maxDepth();

        // save original depth image false color coded
        D_orig.writeColor(prediction_dir + "/depth_orig/" + prefix + ".png", max_depth);

        // save interpolated depth image false color coded
        D_ipol.writeColor(prediction_dir + "/depth_ipol/" + prefix + ".png", max_depth);

        // copy left camera image
        string img_src = gt_img_dir + "/" + prefix + ".png";
        string img_dst = prediction_dir + "/image_0/" + prefix + ".png";
        system(("cp " + img_src + " " + img_dst).c_str());
      }

    // on error, exit
    } catch (...) {
      std::cout << "ERROR: Couldn't read: " << prefix.c_str() << ".png" << std::endl;
      free(namelist_gt);
      free(namelist_prediction);
      return false;
    }
  }
  // open stats file for writing
  string stats_out_file_name = prediction_dir + "/stats_depth.txt";
  FILE *stats_out_file = fopen(stats_out_file_name.c_str(), "w");

  if (stats_out_file == NULL || errors_out.size() == 0) {
    std::cout << "ERROR: Couldn't generate/store output statistics!" << std::endl;
    free(namelist_gt);
    free(namelist_prediction);
    return false;
  }

  const char *metrics[] = {"mae", 
                           "rmse", 
                           "inverse mae", 
                           "inverse rmse", 
                           "log mae", 
                           "log rmse", 
                           "scale invariant log", 
                           "abs relative", 
                           "squared relative"};
  // write mean, min and max
  std::cout << "Done. Your evaluation results are:" << std::endl;
  for (int32_t i = 0; i < errors_out[0].size(); i++) {
    std::cout << "mean " << metrics[i] << ": " << statMean(errors_out, i) << std::endl;
    fprintf(stats_out_file, "mean %s: %f \n", metrics[i], statMean(errors_out, i));
    fprintf(stats_out_file, "min  %s: %f \n", metrics[i], statMin(errors_out, i));
    fprintf(stats_out_file, "max  %s: %f \n", metrics[i], statMax(errors_out, i));
  }

  // close file
  fclose(stats_out_file);
  //free memory of scandir calls
  free(namelist_gt);
  free(namelist_prediction);

  // success
  return true;
}

int32_t main (int32_t argc, char *argv[]) {

  // we need 3 arguments!
  for (int32_t i = 0; i < argc; ++i){
    std::cout << argv[i] << " ";
  }
  std::cout << std::endl;
  if (argc != 3) {
    cout << "Usage: ./evaluate_depth gt_dir prediction_dir" << endl;
    return 1;
  }

  // read arguments
  string gt_img_dir     = argv[1];
  string prediction_dir = argv[2];
  std::cout << "Starting depth evaluation.." << std::endl;
  // run evaluation
  bool success = eval(gt_img_dir, prediction_dir);

  if (success) {
    std::cout << "Your evaluation results are available at:" << std::endl;
    std::cout << prediction_dir + "/stats_depth.txt" << std::endl;
  } else {
    std::cout << "An error occured while processing your results." << std::endl;
    std::cout << "Please make sure that the data in your result directory has the right format (compare to prediction/sparseConv_val)" << std::endl;
  }

  // exit
  return 0;
}

