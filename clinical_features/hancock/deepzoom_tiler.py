import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
import pandas as pd
from unicodedata import normalize
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat
import cv2
from datetime import datetime

Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'

class SlideStats:
    def __init__(self):
        self.total_patches = 0
        self.patches_per_mag = {}
        self.slide_dimensions = None
        self.processing_time = None
        self.background_ratio = None

def generate_thumbnail(slide_path, output_path, max_size=1000):
    """生成WSI缩略图"""
    slide = open_slide(slide_path)
    thumbnail = slide.get_thumbnail((max_size, max_size))
    thumbnail.save(output_path)
    return thumbnail.size

def create_patch_preview(patches_dir, output_path, max_size=2000):
    """创建patch拼接预览图"""
    patches = glob.glob(os.path.join(patches_dir, '*.jpeg'))
    if not patches:
        return
    
    # 读取第一个patch来获取patch大小
    sample_patch = Image.open(patches[0])
    patch_size = sample_patch.size[0]
    
    # 计算网格大小
    n_patches = len(patches)
    grid_size = int(np.ceil(np.sqrt(n_patches)))
    
    # 创建拼接图
    preview = Image.new('RGB', (patch_size * grid_size, patch_size * grid_size), 'white')
    
    for idx, patch_path in enumerate(patches):
        if idx >= grid_size * grid_size:
            break
        row = idx // grid_size
        col = idx % grid_size
        patch = Image.open(patch_path)
        preview.paste(patch, (col * patch_size, row * patch_size))
    
    # 调整大小
    if preview.size[0] > max_size:
        ratio = max_size / preview.size[0]
        new_size = (max_size, int(preview.size[1] * ratio))
        preview = preview.resize(new_size, Image.LANCZOS)
    
    preview.save(output_path)

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                w, h = tile.size
                if edge > self._threshold:
                    if not (w==self._tile_size and h==self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    # tile.save(outfile, quality=self._quality)
                    tile.save(outfile)
            except:
                pass
            self._queue.task_done()
            

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels]
        mag_list = [int(self._mag_base/2**i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            tiledir = os.path.join("%s_files" % self._basename, str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        self.stats = SlideStats()
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20=2, 40/40=0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                    self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def nested_patches(img_slide, out_base, slide_id, level=(0,), ext='jpeg'):
    print('\n Organizing patches')
    # 使用slide_id作为名称，而不是文件名
    img_class = img_slide.split(os.sep)[2]
    n_levels = len(glob.glob('WSI_temp_files/*'))
    bag_path = os.path.join(out_base, img_class, slide_id)
    os.makedirs(bag_path, exist_ok=True)
    
    # 创建统计信息目录
    stats_dir = os.path.join(out_base, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    
    # 生成缩略图
    thumb_dir = os.path.join(out_base, 'thumbnails')
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = os.path.join(thumb_dir, f'{slide_id}_thumbnail.png')
    thumb_size = generate_thumbnail(img_slide, thumb_path)
    
    patch_count = 0
    start_time = datetime.now()
    
    if len(level)==1:
        patches = glob.glob(os.path.join('WSI_temp_files', '*', '*.'+ext))
        patch_count = len(patches)
        
        print(f' Processing {len(patches)} patches...')
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            # sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else:
        level_factor = 2**int(level[1]-level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join('WSI_temp_files', '*'))]
        levels.sort()
        low_patches = glob.glob(os.path.join('WSI_temp_files', str(levels[0]), '*.'+ext))
        print(f' Processing {len(low_patches)} low-resolution patches...')
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(os.path.join('WSI_temp_files', str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext))
                    if len(high_patch)!=0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            try:
                os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            # sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')
    
    # 记录统计信息
    slide = open_slide(img_slide)
    stats = {
        'slide_id': slide_id,  # 使用slide_id而不是img_name
        'slide_name': os.path.basename(img_slide),  # 保留原始文件名作为参考
        'dimensions': slide.dimensions,
        'patch_count': patch_count,
        'magnification_levels': list(level),
        'processing_time': str(datetime.now() - start_time),
        'thumbnail_size': thumb_size,
    }
    
    with open(os.path.join(stats_dir, f'{slide_id}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def read_csv_file_paths(csv_path, base_dir):
    """
    读取CSV文件并返回WSI文件的信息列表
    
    Args:
        csv_path: CSV文件路径
        base_dir: WSI文件的基础目录
    
    Returns:
        List[Tuple[str, str]]: (slide_id, full_path)的元组列表
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件: {csv_path}")
        print(f"CSV包含 {len(df)} 个样本")
        
        # 检查必要的列
        if 'relative_path' not in df.columns:
            raise ValueError("CSV文件中未找到'relative_path'列")
        if 'slide_id' not in df.columns:
            raise ValueError("CSV文件中未找到'slide_id'列")
        
        slide_info = []
        missing_files = []
        
        for idx, row in df.iterrows():
            slide_id = row['slide_id']
            relative_path = row['relative_path']
            full_path = os.path.join(base_dir, relative_path)
            
            if os.path.exists(full_path):
                slide_info.append((slide_id, full_path))
            else:
                missing_files.append((slide_id, relative_path))
                print(f"警告: 文件不存在 - {slide_id}: {relative_path}")
        
        print(f"\n=== 处理摘要 ===")
        print(f"找到 {len(slide_info)} 个有效的WSI文件")
        if missing_files:
            print(f"警告: {len(missing_files)} 个文件不存在")
            print("缺失的文件:")
            for slide_id, relative_path in missing_files[:10]:  # 只显示前10个
                print(f"  - {slide_id}: {relative_path}")
            if len(missing_files) > 10:
                print(f"  ... 还有 {len(missing_files) - 10} 个文件")
        
        return slide_info
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return []

if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('-d', '--dataset', type=str, default='TCGA-lung', help='Dataset name')
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
    parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format for tiles [jpeg]')
    parser.add_argument('-v', '--slide_format', type=str, default='svs', help='Image format for tiles [svs]')
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of worker processes to start [4]')
    parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]')
    parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]')
    parser.add_argument('-b', '--base_mag', type=float, default=20, help='Maximum magnification for patch extraction [20]')
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(0,), help='Levels for patch extraction [0]')
    parser.add_argument('-o', '--objective', type=float, default=20, help='The default objective power if metadata does not present [20]')
    parser.add_argument('-t', '--background_t', type=int, default=15, help='Threshold for filtering background [15]')  
    parser.add_argument('--source_dir', type=str, help='Directory containing WSI files (used when not using CSV)')
    parser.add_argument('--csv_path', type=str, help='Path to CSV file containing relative_path column')
    parser.add_argument('--base_dir', type=str, help='Base directory for WSI files (used with CSV)')
    args = parser.parse_args()
    
    # 验证参数
    if args.csv_path and not args.base_dir:
        parser.error("使用--csv_path时必须指定--base_dir")
    if not args.csv_path and not args.source_dir:
        parser.error("必须指定--csv_path和--base_dir，或者--source_dir")
    
    levels = tuple(sorted(args.magnifications))
    assert len(levels)<=2, 'Only 1 or 2 magnifications are supported!'
    if len(levels) == 2:
        out_base = os.path.join(args.dataset, 'pyramid')
    else:
        out_base = os.path.join(args.dataset, 'single')
    
    # 根据参数选择获取文件列表的方式
    if args.csv_path:
        print("使用CSV文件模式...")
        all_slides_info = read_csv_file_paths(args.csv_path, args.base_dir)
        if not all_slides_info:
            print("错误: 未找到任何WSI文件")
            sys.exit(1)
        
        # pos-i_pos-j -> x, y
        for idx, (slide_id, c_slide) in enumerate(all_slides_info):
            slide_start_time = datetime.now()
            print('Process slide {}/{}: {} (slide_id: {})'.format(idx+1, len(all_slides_info), os.path.basename(c_slide), slide_id))
            
            DeepZoomStaticTiler(c_slide, 'WSI_temp', levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
            nested_patches(c_slide, out_base, slide_id, levels, ext=args.format)
            shutil.rmtree('WSI_temp_files')
            
            total_time = datetime.now() - slide_start_time
            print('Slide completed in: {}\n'.format(total_time))
        
        print('Patch extraction done for {} slides.'.format(len(all_slides_info)))
    else:
        print("使用目录扫描模式...")
        all_slides = glob.glob(os.path.join(args.source_dir, '*.'+args.slide_format))
        
        if not all_slides:
            print("错误: 未找到任何WSI文件")
            sys.exit(1)
        
        # pos-i_pos-j -> x, y
        for idx, c_slide in enumerate(all_slides):
            slide_start_time = datetime.now()
            slide_id = os.path.basename(c_slide).split('.')[0]  # 使用文件名作为slide_id
            print('Process slide {}/{}: {} (slide_id: {})'.format(idx+1, len(all_slides), os.path.basename(c_slide), slide_id))
            
            DeepZoomStaticTiler(c_slide, 'WSI_temp', levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
            nested_patches(c_slide, out_base, slide_id, levels, ext=args.format)
            shutil.rmtree('WSI_temp_files')
            
            total_time = datetime.now() - slide_start_time
            print('Slide completed in: {}\n'.format(total_time))
        
        print('Patch extraction done for {} slides.'.format(len(all_slides)))