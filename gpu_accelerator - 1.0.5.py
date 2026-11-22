"""
gpu_accelerator.py
完整GPU加速Python插件框架（单文件实现）
版本: 1.0.5  # 更新版本号
包含核心功能、插件系统、GUI界面和所有扩展功能
"""

import os
import sys
import json
import time
import inspect
import logging
import threading
import weakref
import atexit
import functools
import hashlib
import pickle
import warnings
from enum import Enum, auto
from typing import (Any, Dict, Optional, Union, List, Tuple, Callable, 
                   TypeVar, Generic, Type, Set, Iterable)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    import numpy as np
except ImportError as e:
    warnings.warn(f"NumPy not available: {str(e)}")
    # 提供虚拟的numpy模块以便非numpy模式运行
    class DummyNumpy:
        ndarray = object
        dtype = object
        def asarray(self, *args, **kwargs): pass
        def empty(self, *args, **kwargs): pass
        def prod(self, *args, **kwargs): pass
    np = DummyNumpy()

try:
    import cupy as cp
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
        QPushButton, QLabel, QComboBox, QTextEdit, QLineEdit, QSpinBox,
        QDoubleSpinBox, QCheckBox, QGroupBox, QFileDialog, QMessageBox,
        QProgressBar, QTreeWidget, QTreeWidgetItem, QSplitter, QDockWidget,
        QTableWidget, QTableWidgetItem, QHeaderView, QMenu, QAction,
        QStatusBar, QToolBar, QSizePolicy, QInputDialog, QScrollArea
    )
    from PyQt5.QtCore import (
        Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize, QPoint,
        QSettings, QFileInfo, QCoreApplication, QByteArray, QObject
    )
    from PyQt5.QtGui import (
        QIcon, QFont, QColor, QPalette, QPixmap, QCursor,
        QTextCursor, QKeySequence
    )
    QT_AVAILABLE = True
except ImportError as e:
    QT_AVAILABLE = False
    warnings.warn(f"GUI dependencies not available: {str(e)}")
    # 提供虚拟类以便非GUI模式运行
    class DummyCuPy:
        ndarray = object
        dtype = object
        
        class cuda:
            class Device:
                def __init__(self, device_id=None):
                    self.id = device_id if device_id is not None else 0
                
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
                
                def use(self):
                    pass
                
                def synchronize(self):
                    pass
                
                @property
                def compute_capability(self):
                    return (0, 0)
                
                @property
                def name(self):
                    return "CPU Fallback Device"
                
                @property
                def total_memory(self):
                    return 0
                
                def retain_primary_context(self):
                    return self
            
            class Stream:
                def __init__(self, non_blocking=False):
                    pass
                
                def synchronize(self):
                    pass
            
            class Event:
                def __init__(self, disable_timing=False):
                    pass
                
                def synchronize(self):
                    pass
            
            class runtime:
                @staticmethod
                def memGetInfo():
                    return (0, 0)
                
                @staticmethod
                def getDeviceCount():
                    return 0
        
        @staticmethod
        def asarray(arr, stream=None):
            return arr
        
        @staticmethod
        def asnumpy(arr):
            return arr
        
        @staticmethod
        def empty(shape, dtype=None):
            return np.empty(shape, dtype=dtype)
        
        @staticmethod
        def get_default_memory_pool():
            return DummyMemoryPool()
        
        @staticmethod
        def get_default_pinned_memory_pool():
            return DummyMemoryPool()
        
        class core:
            @staticmethod
            def set_precision(*args):
                pass

    class DummyMemoryPool:
        def free_all_blocks(self):
            pass
        
        def malloc(self, size):
            return bytearray(size)
        
        def free(self, ptr):
            pass

    cp = DummyCuPy()

    
    class QObject: pass
    class pyqtSignal: pass

# ==================== 核心类型和常量 ====================

T = TypeVar('T')
ArrayLike = Union[np.ndarray, cp.ndarray] if hasattr(cp, 'ndarray') else np.ndarray
PluginConfig = Dict[str, Any]
MemoryInfo = Tuple[int, int, int]  # 已用, 空闲, 总量

DEFAULT_DEVICE_ID = 0
MAX_WORKERS = 4
MEMORY_CHECK_INTERVAL = 0.1  # 秒
CACHE_SIZE = 1024 * 1024 * 1024  # 1GB缓存

# ==================== 枚举类 ====================

class PrecisionMode(Enum):
    """计算精度模式"""
    FP32 = auto()
    FP16 = auto()
    TF32 = auto()
    MIXED = auto()

class PluginState(Enum):
    """插件状态"""
    INITIALIZED = auto()
    READY = auto()
    BUSY = auto()
    ERROR = auto()
    SHUTDOWN = auto()

# ==================== 数据类 ====================

@dataclass
class BenchmarkResult:
    """性能测试结果"""
    time_elapsed: float
    memory_used: int
    throughput: float
    device_info: Dict[str, Any]
    precision: PrecisionMode
    timestamp: float

# ==================== 异常类 ====================

class GPUPluginError(Exception):
    """基础插件异常"""
    pass

class MemoryAllocationError(GPUPluginError):
    """内存分配异常"""
    pass

class DeviceUnavailableError(GPUPluginError):
    """设备不可用异常"""
    pass

class PluginConfigurationError(GPUPluginError):
    """配置错误"""
    pass

class PluginDependencyError(GPUPluginError):
    """依赖错误"""
    pass

# ==================== 核心实现 ====================

class _MemoryPool:
    """自定义内存池实现"""
    
    def __init__(self, max_size: int = CACHE_SIZE):
        self._max_size = max_size
        self._used = 0
        self._pool: Dict[bytes, cp.ndarray] = {}
        self._lock = threading.Lock()
        
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> cp.ndarray:
        """分配内存"""
        key = self._make_key(shape, dtype)
        
        with self._lock:
            if key in self._pool:
                return self._pool.pop(key)
            
            # 检查内存限制
            size = np.prod(shape) * np.dtype(dtype).itemsize
            if self._used + size > self._max_size:
                self._cleanup(size)
                
            arr = cp.empty(shape, dtype=dtype)
            self._used += arr.nbytes
            return arr
    
    def deallocate(self, array: cp.ndarray) -> None:
        """释放内存到池中"""
        if array is None:
            return
            
        key = self._make_key(array.shape, array.dtype)
        
        with self._lock:
            if key in self._pool or self._used + array.nbytes > self._max_size:
                array = None  # 显式释放
            else:
                self._pool[key] = array
                self._used += array.nbytes
    
    def _make_key(self, shape: Tuple[int, ...], dtype: np.dtype) -> bytes:
        """生成内存键"""
        return hashlib.sha256(pickle.dumps((shape, dtype))).digest()
    
    def _cleanup(self, required_size: int) -> None:
        """清理内存"""
        while self._pool and self._used > self._max_size - required_size:
            _, arr = self._pool.popitem()
            self._used -= arr.nbytes
            arr = None
    
    def clear(self) -> None:
        """清空内存池"""
        with self._lock:
            self._pool.clear()
            self._used = 0

class GPUPluginBase:
    """完整的GPU插件基类"""
    
    _instances = weakref.WeakSet()
    _memory_pool = _MemoryPool()
    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    def __init__(self, 
                 device_id: int = DEFAULT_DEVICE_ID,
                 config: Optional[PluginConfig] = None):
        """
        初始化GPU插件
        
        参数:
            device_id: 使用的GPU设备ID
            config: 插件配置字典
        """
        self._device_id = device_id
        self._state = PluginState.INITIALIZED
        self._stream = None
        self._event = None
        self._memory_tracker = {}
        self._benchmark_results = []
        self._call_history = []
        self._lock = threading.Lock()
        self._config = self._init_config(config)
        self._precision = PrecisionMode.FP32
        self._context = None
        
        self._init_device()
        self._init_stream()
        self._register_instance()
        
    def _init_config(self, config: Optional[PluginConfig]) -> PluginConfig:
        """初始化配置"""
        default_config = {
            "max_memory_usage": 0.9,
            "auto_cleanup": True,
            "enable_cache": True,
            "async_operations": False,
            "log_level": "INFO",
            "profile_performance": False,
            "fallback_to_cpu": False,
            "memory_check_interval": MEMORY_CHECK_INTERVAL,
        }
        
        if config:
            default_config.update(config)
            
        # 验证配置
        if not 0 < default_config["max_memory_usage"] <= 1:
            raise PluginConfigurationError("max_memory_usage必须在(0,1]范围内")
            
        # 设置日志级别
        logging.basicConfig(level=getattr(logging, default_config["log_level"]))
            
        return default_config
    
    def _init_device(self) -> None:
        """初始化GPU设备"""
        try:
            if isinstance(cp, DummyCuPy):
                # CPU回退模式
                self._context = None
                self._state = PluginState.READY
                logging.warning("运行在CPU回退模式")
                return
                
            self._context = cp.cuda.Device(self._device_id).retain_primary_context()
            self._context.push()
            
            # 检查计算能力
            major, minor = cp.cuda.Device(self._device_id).compute_capability
            if major < 6:
                warnings.warn(f"GPU设备{self._device_id}计算能力较低({major}.{minor})")
                
            # 设置默认精度
            if self._config.get("enable_tf32", False) and major >= 8:
                self._precision = PrecisionMode.TF32
                cp.core.set_precision('float32', 'tf32')
            elif self._config.get("enable_fp16", False) and minor >= 0:
                self._precision = PrecisionMode.FP16
            
            self._state = PluginState.READY
            logging.info(f"GPU设备{self._device_id}初始化完成, 计算能力: {major}.{minor}")
            
        except Exception as e:
            self._state = PluginState.ERROR
            logging.error(f"GPU设备{self._device_id}初始化失败: {str(e)}")
            if self._config["fallback_to_cpu"]:
                logging.warning("将回退到CPU模式")
                self._state = PluginState.READY
            else:
                raise DeviceUnavailableError(f"无法初始化GPU设备{self._device_id}") from e

    
    def _init_stream(self) -> None:
        """初始化CUDA流"""
        if self._state == PluginState.ERROR:
            return
            
        try:
            if isinstance(cp, DummyCuPy):
                # CPU回退模式
                self._stream = cp.cuda.Stream()
                self._event = cp.cuda.Event()
                return
                
            self._stream = cp.cuda.Stream(non_blocking=True)
            self._event = cp.cuda.Event(disable_timing=False)
            logging.debug(f"GPU设备{self._device_id}流初始化完成")
        except Exception as e:
            self._state = PluginState.ERROR
            logging.error(f"无法创建CUDA流: {str(e)}")
            if self._config["fallback_to_cpu"]:
                logging.warning("将回退到CPU模式")
                self._stream = cp.cuda.Stream()
                self._event = cp.cuda.Event()
                self._state = PluginState.READY
            else:
                raise DeviceUnavailableError(f"无法创建CUDA流") from e

    
    def _register_instance(self) -> None:
        """注册实例以便全局管理"""
        GPUPluginBase._instances.add(self)
        atexit.register(self.cleanup)
    
    @property
    def state(self) -> PluginState:
        """获取当前插件状态"""
        return self._state
    
    @property
    def device_id(self) -> int:
        """获取设备ID"""
        return self._device_id
    
    @property
    def precision(self) -> PrecisionMode:
        """获取当前计算精度"""
        return self._precision
    
    @precision.setter
    def precision(self, mode: PrecisionMode) -> None:
        """设置计算精度"""
        if self._state != PluginState.READY:
            raise GPUPluginError("插件未就绪，无法更改精度模式")
            
        if mode == PrecisionMode.TF32:
            cp.core.set_precision('float32', 'tf32')
        elif mode == PrecisionMode.FP16:
            cp.core.set_precision('float16')
        else:
            cp.core.set_precision('float32')
        
        self._precision = mode
        logging.info(f"精度模式设置为: {mode.name}")
    
    def set_config(self, key: str, value: Any) -> None:
        """更新单个配置项"""
        with self._lock:
            if key in self._config:
                old_value = self._config[key]
                self._config[key] = value
                logging.info(f"配置更新: {key}={old_value} -> {value}")
                
                # 特殊配置需要额外处理
                if key == "max_memory_usage":
                    if not 0 < value <= 1:
                        raise PluginConfigurationError("max_memory_usage必须在(0,1]范围内")
                elif key == "log_level":
                    logging.getLogger().setLevel(value.upper())
            else:
                raise PluginConfigurationError(f"无效配置项: {key}")
    
    def update_config(self, new_config: PluginConfig) -> None:
        """批量更新配置"""
        with self._lock:
            for key, value in new_config.items():
                self.set_config(key, value)
    
    def get_memory_info(self) -> MemoryInfo:
        """获取内存使用信息"""
        if self._state == PluginState.ERROR:
            return (0, 0, 0)
            
        if isinstance(cp, DummyCuPy):
            # CPU回退模式，返回虚拟内存信息
            return (0, 2**31, 2**31)  # 2GB虚拟内存
            
        free, total = cp.cuda.runtime.memGetInfo()
        used = total - free
        return (used, free, total)
    
    def check_memory(self, required_bytes: int, timeout: float = 5.0) -> bool:
        """
        检查是否有足够内存，必要时等待
        
        参数:
            required_bytes: 需要的内存大小(字节)
            timeout: 最大等待时间(秒)
            
        返回:
            bool: 是否有足够内存
        """
        if self._state == PluginState.ERROR:
            return False
            
        if isinstance(cp, DummyCuPy):
            # CPU回退模式，总是返回True
            return True
            
        start_time = time.time()
        interval = self._config["memory_check_interval"]
        
        while True:
            used, free, total = self.get_memory_info()
            available = free * self._config["max_memory_usage"]
            
            if required_bytes <= available:
                return True
                
            if time.time() - start_time > timeout:
                logging.warning(f"内存不足: 需要{required_bytes}, 可用{available}")
                return False
                
            time.sleep(interval)
    
    def _track_memory(self, name: str, array: cp.ndarray) -> None:
        """跟踪内存分配"""
        with self._lock:
            self._memory_tracker[name] = {
                "shape": array.shape,
                "dtype": array.dtype,
                "size": array.nbytes,
                "time": time.time(),
                "stack": inspect.stack()[1:]
            }
    
    def memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        with self._lock:
            used, free, total = self.get_memory_info()
            
            return {
                "total": total,
                "used": used,
                "free": free,
                "usage_percent": used / total * 100,
                "tracked_objects": len(self._memory_tracker),
                "tracked_memory": sum(v["size"] for v in self._memory_tracker.values()),
                "objects": self._memory_tracker
            }
    
    def _async_operation(self, func: Callable[..., T]) -> Callable[..., T]:
        """装饰器：将操作转为异步执行"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._config["async_operations"]:
                return func(*args, **kwargs)
                
            self._state = PluginState.BUSY
            
            try:
                future = self._executor.submit(func, *args, **kwargs)
                result = future.result()
                self._state = PluginState.READY
                return result
            except Exception as e:
                self._state = PluginState.ERROR
                logging.error(f"异步操作失败: {str(e)}")
                raise
                
        return wrapper
    
    def _profile_performance(self, func: Callable[..., T]) -> Callable[..., T]:
        """装饰器：性能分析"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._config["profile_performance"]:
                return func(*args, **kwargs)
                
            # 预热
            func(*args, **kwargs)
            self.synchronize()
            
            # 记录内存使用前
            mem_before = self.get_memory_info()[0]
            
            # 执行并计时
            start_time = time.time()
            result = func(*args, **kwargs)
            self.synchronize()
            elapsed = time.time() - start_time
            
            # 记录内存使用后
            mem_after = self.get_memory_info()[0]
            
            # 计算吞吐量 (假设操作是矩阵乘法)
            throughput = 0
            if len(args) >= 2 and hasattr(args[0], 'shape') and hasattr(args[1], 'shape'):
                m, n = args[0].shape[0], args[1].shape[1]
                ops = 2 * m * n * args[0].shape[1]  # 2MNK flops
                throughput = ops / elapsed / 1e9  # GFLOPs
            
            # 保存结果
            benchmark_result = BenchmarkResult(
                time_elapsed=elapsed,
                memory_used=mem_after - mem_before,
                throughput=throughput,
                device_info=self.device_info,
                precision=self.precision,
                timestamp=time.time()
            )
            
            with self._lock:
                self._benchmark_results.append(benchmark_result)
                self._call_history.append({
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "timestamp": time.time(),
                    "result": str(result)[:100]  # 只存储部分结果
                })
            
            return result
            
        return wrapper
    
    @property
    def device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if self._state == PluginState.ERROR:
            return {}
            
        dev = cp.cuda.Device(self._device_id)
        attrs = [
            'name', 'total_memory', 'multi_processor_count',
            'max_threads_per_block', 'max_block_dim_x',
            'max_grid_dim_x', 'max_shared_memory_per_block'
        ]
        
        info = {attr: getattr(dev, attr, None) for attr in attrs}
        info['compute_capability'] = dev.compute_capability
        info['uuid'] = dev.uuid if hasattr(dev, 'uuid') else None
        info['pci_bus_id'] = dev.pci_bus_id if hasattr(dev, 'pci_bus_id') else None
        
        return info
    
    def synchronize(self) -> None:
        """同步设备"""
        if self._state not in (PluginState.ERROR, PluginState.SHUTDOWN):
            self._stream.synchronize()
    
    def cleanup(self, full: bool = False) -> None:
        """
        清理资源
        
        参数:
            full: 是否完全清理(包括内存池)
        """
        if self._state == PluginState.SHUTDOWN:
            return
            
        self.synchronize()
        
        # 清理内存
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        if full:
            self._memory_pool.clear()
        
        with self._lock:
            self._memory_tracker.clear()
        
        self._state = PluginState.READY
        logging.info(f"GPU设备{self._device_id}资源已清理")
    
    def reset(self) -> None:
        """重置插件状态"""
        if self._state == PluginState.SHUTDOWN:
            raise GPUPluginError("插件已关闭，无法重置")
            
        self.cleanup(full=True)
        self._init_device()
        self._init_stream()
        self._state = PluginState.READY
    
    def __del__(self) -> None:
        """析构函数"""
        if self._state != PluginState.SHUTDOWN:
            self.shutdown()
    
    def shutdown(self) -> None:
        """关闭插件并释放所有资源"""
        if self._state == PluginState.SHUTDOWN:
            return
            
        self.cleanup(full=True)
        
        if self._context:
            self._context.pop()
            self._context = None
            
        self._state = PluginState.SHUTDOWN
        logging.info(f"GPU设备{self._device_id}插件已关闭")
    
    @staticmethod
    def to_gpu(array: np.ndarray, stream=None) -> cp.ndarray:
        """传输数据到GPU"""
        return cp.asarray(array, stream=stream)
    
    @staticmethod
    def to_cpu(array: cp.ndarray, stream=None) -> np.ndarray:
        """传输数据回CPU"""
        return cp.asnumpy(array, stream=stream)
    
    def benchmark(self, func: Callable[..., T], *args, **kwargs) -> BenchmarkResult:
        """
        对函数进行基准测试
        
        参数:
            func: 要测试的函数
            *args, **kwargs: 函数参数
            
        返回:
            BenchmarkResult: 测试结果
        """
        # 临时启用性能分析
        original_profile = self._config["profile_performance"]
        self._config["profile_performance"] = True
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 恢复配置
        self._config["profile_performance"] = original_profile
        
        # 返回最后一次结果
        with self._lock:
            if self._benchmark_results:
                return self._benchmark_results[-1]
            return BenchmarkResult(0, 0, 0, {}, self.precision, time.time())
    
    def get_benchmark_results(self, clear: bool = False) -> List[BenchmarkResult]:
        """获取所有性能测试结果"""
        with self._lock:
            results = self._benchmark_results.copy()
            if clear:
                self._benchmark_results.clear()
            return results
    
    def get_call_history(self, clear: bool = False) -> List[Dict[str, Any]]:
        """获取调用历史"""
        with self._lock:
            history = self._call_history.copy()
            if clear:
                self._call_history.clear()
            return history
    
    @classmethod
    def get_all_instances(cls) -> List['GPUPluginBase']:
        """获取所有活动实例"""
        return list(cls._instances)
    
    @classmethod
    def shutdown_all(cls) -> None:
        """关闭所有插件实例"""
        for instance in cls.get_all_instances():
            instance.shutdown()

class PluginManager:
    """高级插件管理器"""
    
    def __init__(self):
        self._plugins: Dict[str, Type[GPUPluginBase]] = {}
        self._plugin_dependencies: Dict[str, Set[str]] = {}
        self._plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self._device_status = self._scan_devices()
        self._lock = threading.Lock()
        
        # 自动注册内置插件
        self._register_builtin_plugins()
    
    def _scan_devices(self) -> Dict[int, Dict[str, Any]]:
        """扫描可用GPU设备"""
        devices = {}
        try:
            count = cp.cuda.runtime.getDeviceCount()
            if count == 0:
                logging.warning("没有检测到GPU设备")
                return devices
                
            for i in range(count):
                try:
                    dev = cp.cuda.Device(i)
                    devices[i] = {
                        "name": dev.name,
                        "total_memory": dev.total_memory,
                        "capability": dev.compute_capability,
                        "available": True
                    }
                    logging.info(f"检测到GPU设备 {i}: {devices[i]['name']}")
                except Exception as e:
                    logging.warning(f"无法初始化GPU设备 {i}: {str(e)}")
                    devices[i] = {
                        "name": f"Unknown Device {i}",
                        "total_memory": 0,
                        "capability": (0, 0),
                        "available": False
                    }
        except Exception as e:
            logging.error(f"检测GPU设备时发生错误: {str(e)}")
            # 添加一个虚拟CPU设备作为回退
            devices[0] = {
                "name": "CPU Fallback",
                "total_memory": 0,
                "capability": (0, 0),
                "available": True
            }
        return devices

    
    def _register_builtin_plugins(self) -> None:
        """注册内置插件"""
        # 在实际实现中，这里会动态发现并注册插件
        pass
    
    def register_plugin(self, 
                       name: str, 
                       plugin_class: Type[GPUPluginBase],
                       dependencies: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        注册插件
        
        参数:
            name: 插件名称
            plugin_class: 插件类
            dependencies: 依赖的其他插件名
            metadata: 插件元数据
        """
        if not inspect.isclass(plugin_class) or not issubclass(plugin_class, GPUPluginBase):
            raise PluginConfigurationError("插件必须是GPUPluginBase的子类")
            
        with self._lock:
            self._plugins[name] = plugin_class
            self._plugin_dependencies[name] = set(dependencies or [])
            self._plugin_metadata[name] = metadata or {}
            logging.info(f"插件 '{name}' 已注册")
    
    def unregister_plugin(self, name: str) -> None:
        """注销插件"""
        with self._lock:
            if name in self._plugins:
                del self._plugins[name]
                del self._plugin_dependencies[name]
                del self._plugin_metadata[name]
                logging.info(f"插件 '{name}' 已注销")
            else:
                logging.warning(f"尝试注销未注册的插件: {name}")
    
    def create_instance(self, 
                       name: str, 
                       device_id: Optional[int] = None,
                       config: Optional[PluginConfig] = None,
                       check_dependencies: bool = True) -> GPUPluginBase:
        """
        创建插件实例
        
        参数:
            name: 插件名称
            device_id: 使用的设备ID
            config: 插件配置
            check_dependencies: 是否检查依赖
            
        返回:
            GPUPluginBase: 插件实例
        """
        if name not in self._plugins:
            raise PluginDependencyError(f"未注册的插件: {name}")
            
        # 检查依赖
        if check_dependencies:
            missing = self._check_missing_dependencies(name)
            if missing:
                raise PluginDependencyError(f"插件 '{name}' 缺少依赖: {missing}")
        
        # 自动选择设备
        if device_id is None:
            device_id = self._select_best_device(name)
        
        # 确保配置中包含回退到CPU的选项
        if config is None:
            config = {}
        config.setdefault("fallback_to_cpu", True)
        
        try:
            plugin = self._plugins[name](device_id=device_id, config=config)
            logging.info(f"插件 '{name}' 实例已创建于设备 {device_id}")
            return plugin
        except DeviceUnavailableError as e:
            # 如果没有GPU设备，尝试使用CPU回退
            if device_id != 0:  # 已经尝试过设备0
                raise
            
            logging.warning(f"无法在设备 {device_id} 上创建插件，尝试CPU回退模式")
            config["fallback_to_cpu"] = True
            try:
                plugin = self._plugins[name](device_id=device_id, config=config)
                logging.warning(f"插件 '{name}' 在CPU回退模式下运行")
                return plugin
            except Exception as e:
                logging.error(f"创建插件 '{name}' 实例失败: {str(e)}")
                raise
        except Exception as e:
            logging.error(f"创建插件 '{name}' 实例失败: {str(e)}")
            raise

    
    def _check_missing_dependencies(self, plugin_name: str) -> Set[str]:
        """检查缺失的依赖"""
        required = self._plugin_dependencies.get(plugin_name, set())
        return required - set(self._plugins.keys())
    
    def _select_best_device(self, plugin_name: str) -> int:
        """为插件选择最佳设备"""
        # 优先选择可用的GPU设备
        for dev_id, status in self._device_status.items():
            if status["available"]:
                return dev_id
        
        # 如果没有可用的GPU设备，返回0作为回退
        logging.warning("没有可用的GPU设备，将使用设备0作为回退")
        return 0

    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """列出所有注册的插件"""
        with self._lock:
            return {
                name: {
                    "class": plugin_class.__name__,
                    "dependencies": list(self._plugin_dependencies[name]),
                    **self._plugin_metadata[name]
                }
                for name, plugin_class in self._plugins.items()
            }
    
    def list_devices(self) -> Dict[int, Dict[str, Any]]:
        """列出所有可用设备"""
        return self._device_status.copy()
    
    def get_plugin_info(self, name: str) -> Dict[str, Any]:
        """获取插件详细信息"""
        with self._lock:
            if name not in self._plugins:
                raise PluginDependencyError(f"未注册的插件: {name}")
                
            info = {
                "name": name,
                "class": self._plugins[name].__name__,
                "doc": inspect.getdoc(self._plugins[name]),
                "dependencies": list(self._plugin_dependencies[name]),
                **self._plugin_metadata[name]
            }
            
            # 添加类方法信息
            methods = {}
            for meth_name, meth in inspect.getmembers(self._plugins[name], inspect.isfunction):
                if not meth_name.startswith('_'):
                    methods[meth_name] = {
                        "doc": inspect.getdoc(meth),
                        "signature": str(inspect.signature(meth))
                    }
            
            info["methods"] = methods
            return info

# ==================== GUI实现 ====================

if QT_AVAILABLE:
    class WorkerSignals(QObject):
        """工作线程信号"""
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        progress = pyqtSignal(int)
        message = pyqtSignal(str)
        memory_usage = pyqtSignal(int, int, int)  # 已用, 空闲, 总量

    class PluginWorker(QThread):
        """插件工作线程"""
        
        def __init__(self, plugin: GPUPluginBase, method: str, *args, **kwargs):
            super().__init__()
            self.plugin = plugin
            self.method = method
            self.args = args
            self.kwargs = kwargs
            self.signals = WorkerSignals()
            self._stop_requested = False
            self._progress = 0
        
        def run(self):
            """执行插件方法"""
            try:
                # 获取方法
                func = getattr(self.plugin, self.method)
                
                # 如果是长时间运行的方法，定期报告进度
                if hasattr(func, '__is_long_running__'):
                    result = func(
                        *self.args, 
                        progress_callback=self._progress_callback,
                        **self.kwargs
                    )
                else:
                    result = func(*self.args, **self.kwargs)
                    
                self.signals.finished.emit(result)
            except Exception as e:
                self.signals.error.emit(str(e))
        
        def _progress_callback(self, progress: int):
            """进度回调"""
            self._progress = progress
            self.signals.progress.emit(progress)
            
            # 同时报告内存使用情况
            used, free, total = self.plugin.get_memory_info()
            self.signals.memory_usage.emit(used, free, total)
            
            if self._stop_requested:
                raise GPUPluginError("操作已取消")
        
        def stop(self):
            """请求停止"""
            self._stop_requested = True

    class PluginDockWidget(QDockWidget):
        """插件停靠窗口基类"""
        
        def __init__(self, title: str, parent=None):
            super().__init__(title, parent)
            self.setFeatures(
                QDockWidget.DockWidgetMovable |
                QDockWidget.DockWidgetFloatable
            )
            self._setup_ui()
        
        def _setup_ui(self):
            """初始化UI"""
            self.widget = QWidget()
            self.layout = QVBoxLayout()
            self.widget.setLayout(self.layout)
            self.setWidget(self.widget)
        
        def add_action(self, name: str, callback, shortcut=None, icon=None):
            """添加动作按钮"""
            btn = QPushButton(name)
            if icon:
                btn.setIcon(QIcon(icon))
            if shortcut:
                btn.setShortcut(QKeySequence(shortcut))
            btn.clicked.connect(callback)
            self.layout.addWidget(btn)
            return btn

    class MemoryMonitorDock(PluginDockWidget):
        """内存监控停靠窗口"""
        
        def __init__(self, parent=None):
            super().__init__("内存监控", parent)
            self._setup_memory_ui()
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_memory_usage)
            self.timer.start(1000)  # 每秒更新
        
        def _setup_memory_ui(self):
            """设置内存监控UI"""
            # 内存使用进度条
            self.memory_bar = QProgressBar()
            self.memory_bar.setRange(0, 100)
            self.memory_bar.setFormat("GPU内存使用: %p%")
            self.layout.addWidget(self.memory_bar)
            
            # 内存详细信息
            self.memory_info = QLabel()
            self.memory_info.setWordWrap(True)
            self.layout.addWidget(self.memory_info)
            
            # 内存历史图表占位
            self.memory_plot = QLabel("内存使用历史图表将显示在这里")
            self.memory_plot.setAlignment(Qt.AlignCenter)
            self.memory_plot.setMinimumHeight(150)
            self.layout.addWidget(self.memory_plot)
        
        def update_memory_usage(self, used: int = None, free: int = None, total: int = None):
            """更新内存使用显示"""
            if used is None or free is None or total is None:
                # 如果没有提供数据，从插件获取
                plugin = self.parent().current_plugin
                if plugin and plugin.state == PluginState.READY:
                    used, free, total = plugin.get_memory_info()
                else:
                    return
            
            percent = used / total * 100
            self.memory_bar.setValue(int(percent))
            
            info_text = (
                f"已用: {used / 1024**2:.2f} MB\n"
                f"可用: {free / 1024**2:.2f} MB\n"
                f"总量: {total / 1024**2:.2f} MB"
            )
            self.memory_info.setText(info_text)

    class PluginConsoleDock(PluginDockWidget):
        """插件控制台停靠窗口"""
        
        def __init__(self, parent=None):
            super().__init__("控制台", parent)
            self._setup_console_ui()
            self.history = []
            self.history_index = -1
        
        def _setup_console_ui(self):
            """设置控制台UI"""
            # 输入框
            self.input = QLineEdit()
            self.input.setPlaceholderText("输入插件命令...")
            self.input.returnPressed.connect(self.execute_command)
            self.layout.addWidget(self.input)
            
            # 输出区域
            self.output = QTextEdit()
            self.output.setReadOnly(True)
            self.output.setFont(QFont("Courier New", 10))
            self.layout.addWidget(self.output)
            
            # 按钮
            btn_layout = QHBoxLayout()
            self.execute_btn = QPushButton("执行")
            self.execute_btn.clicked.connect(self.execute_command)
            btn_layout.addWidget(self.execute_btn)
            
            self.clear_btn = QPushButton("清除")
            self.clear_btn.clicked.connect(self.output.clear)
            btn_layout.addWidget(self.clear_btn)
            
            self.layout.addLayout(btn_layout)
        
        def execute_command(self):
            """执行控制台命令"""
            command = self.input.text().strip()
            if not command:
                return
                
            self.append_output(f">>> {command}")
            self.history.append(command)
            self.history_index = len(self.history)
            self.input.clear()
            
            try:
                # 简单的命令解析
                if command.lower() == "help":
                    self.show_help()
                elif command.lower().startswith("list"):
                    self.handle_list_command(command)
                else:
                    # 尝试作为Python表达式执行
                    result = self.evaluate_command(command)
                    self.append_output(str(result))
            except Exception as e:
                self.append_output(f"错误: {str(e)}", error=True)
        
        def evaluate_command(self, command: str):
            """评估命令"""
            plugin = self.parent().current_plugin
            if not plugin:
                raise GPUPluginError("没有活动的插件")
                
            # 安全地评估命令
            allowed_names = {
                'plugin': plugin,
                'np': np,
                'math': __import__('math')
            }
            
            # 编译代码
            code = compile(command, "<string>", "eval")
            
            # 检查允许的名称
            for name in code.co_names:
                if name not in allowed_names:
                    raise NameError(f"名称 '{name}' 未定义")
                    
            return eval(code, {"__builtins__": {}}, allowed_names)
        
        def append_output(self, text: str, error: bool = False):
            """追加输出"""
            cursor = self.output.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            if error:
                self.output.setTextColor(QColor(255, 0, 0))
            else:
                self.output.setTextColor(QColor(0, 0, 0))
                
            cursor.insertText(text + "\n")
            self.output.setTextCursor(cursor)
            self.output.ensureCursorVisible()
        
        def show_help(self):
            """显示帮助信息"""
            help_text = (
                "可用命令:\n"
                "help - 显示此帮助\n"
                "list plugins - 列出所有插件\n"
                "list devices - 列出所有设备\n"
                "plugin.method(args) - 调用插件方法\n"
                "数学表达式 - 例如: np.array([1,2,3])"
            )
            self.append_output(help_text)
        
        def handle_list_command(self, command: str):
            """处理list命令"""
            parts = command.split()
            if len(parts) < 2:
                self.append_output("用法: list [plugins|devices]")
                return
                
            manager = self.parent().plugin_manager
            if parts[1] == "plugins":
                plugins = manager.list_plugins()
                self.append_output(json.dumps(plugins, indent=2))
            elif parts[1] == "devices":
                devices = manager.list_devices()
                self.append_output(json.dumps(devices, indent=2))
            else:
                self.append_output(f"未知列表类型: {parts[1]}")

    class BenchmarkResultsDock(PluginDockWidget):
        """性能测试结果停靠窗口"""
        
        def __init__(self, parent=None):
            super().__init__("性能测试", parent)
            self._setup_benchmark_ui()
        
        def _setup_benchmark_ui(self):
            """设置性能测试UI"""
            # 结果表格
            self.results_table = QTableWidget()
            self.results_table.setColumnCount(6)
            self.results_table.setHorizontalHeaderLabels([
                "方法", "时间(ms)", "内存(MB)", "吞吐量", "设备", "时间戳"
            ])
            self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.layout.addWidget(self.results_table)
            
            # 控制按钮
            btn_layout = QHBoxLayout()
            self.export_btn = QPushButton("导出")
            self.export_btn.clicked.connect(self.export_results)
            btn_layout.addWidget(self.export_btn)
            
            self.clear_btn = QPushButton("清除")
            self.clear_btn.clicked.connect(self.clear_results)
            btn_layout.addWidget(self.clear_btn)
            
            self.layout.addLayout(btn_layout)
        
        def update_results(self, results: List[BenchmarkResult]):
            """更新性能测试结果"""
            self.results_table.setRowCount(len(results))
            
            for row, result in enumerate(results):
                self.results_table.setItem(row, 0, QTableWidgetItem(result.function))
                self.results_table.setItem(row, 1, QTableWidgetItem(f"{result.time_elapsed*1000:.2f}"))
                self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.memory_used/1024**2:.2f}"))
                self.results_table.setItem(row, 3, QTableWidgetItem(f"{result.throughput:.2f} GFLOPs" if result.throughput > 0 else "N/A"))
                self.results_table.setItem(row, 4, QTableWidgetItem(result.device_info.get('name', 'Unknown')))
                self.results_table.setItem(row, 5, QTableWidgetItem(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result.timestamp))))
        
        def export_results(self):
            """导出结果"""
            filename, _ = QFileDialog.getSaveFileName(
                self, "导出性能测试结果", "", "CSV文件 (*.csv);;JSON文件 (*.json)")
            
            if not filename:
                return
                
            if filename.endswith('.csv'):
                self._export_to_csv(filename)
            elif filename.endswith('.json'):
                self._export_to_json(filename)
            else:
                QMessageBox.warning(self, "导出错误", "不支持的格式")
        
        def _export_to_csv(self, filename: str):
            """导出为CSV"""
            try:
                with open(filename, 'w') as f:
                    f.write("method,time_ms,memory_mb,throughput,device,timestamp\n")
                    for row in range(self.results_table.rowCount()):
                        items = [
                            self.results_table.item(row, col).text()
                            for col in range(self.results_table.columnCount())
                        ]
                        f.write(','.join(items) + '\n')
                QMessageBox.information(self, "导出成功", f"结果已保存到 {filename}")
            except Exception as e:
                QMessageBox.critical(self, "导出错误", f"无法保存文件: {str(e)}")
        
        def _export_to_json(self, filename: str):
            """导出为JSON"""
            try:
                results = []
                for row in range(self.results_table.rowCount()):
                    results.append({
                        "method": self.results_table.item(row, 0).text(),
                        "time_ms": float(self.results_table.item(row, 1).text()),
                        "memory_mb": float(self.results_table.item(row, 2).text()),
                        "throughput": self.results_table.item(row, 3).text(),
                        "device": self.results_table.item(row, 4).text(),
                        "timestamp": self.results_table.item(row, 5).text()
                    })
                
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                QMessageBox.information(self, "导出成功", f"结果已保存到 {filename}")
            except Exception as e:
                QMessageBox.critical(self, "导出错误", f"无法保存文件: {str(e)}")
        
        def clear_results(self):
            """清除结果"""
            self.results_table.setRowCount(0)

    class PluginMainWindow(QMainWindow):
        """GPU插件主窗口"""
        
        def __init__(self):
            super().__init__()
            
            # 初始化管理器
            self.plugin_manager = PluginManager()
            self.current_plugin = None
            self.workers = []
            
            # 应用设置
            self.settings = QSettings("GPUPlugin", "GPUAcceleratedPlugin")
            self._load_settings()
            
            # 初始化UI
            self.init_ui()
            self.setWindowTitle("GPU加速插件 - 完整版")
            self.setWindowIcon(QIcon(self._resource_path("icon.png")))
            self.showMaximized()
            
            # 注册内置插件
            self._register_builtin_plugins()
        
        def _resource_path(self, relative_path: str) -> str:
            """获取资源路径"""
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base_path, relative_path)
        
        def _load_settings(self):
            """加载设置"""
            self.restoreGeometry(self.settings.value("windowGeometry", QByteArray()))
            self.restoreState(self.settings.value("windowState", QByteArray()))
        
        def _save_settings(self):
            """保存设置"""
            self.settings.setValue("windowGeometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())
        
        def _register_builtin_plugins(self):
            """注册内置插件"""
            # 在实际应用中，这里会动态加载插件模块
            pass
        
        def init_ui(self):
            """初始化用户界面"""
            # 主窗口设置
            self.setMinimumSize(800, 600)
            
            # 中央部件
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout()
            central_widget.setLayout(main_layout)
            
            # 菜单栏
            self._setup_menu_bar()
            
            # 工具栏
            self._setup_tool_bar()
            
            # 主标签页
            self.tab_widget = QTabWidget()
            main_layout.addWidget(self.tab_widget)
            
            # 添加插件标签页
            self._setup_plugin_tabs()
            
            # 添加停靠窗口
            self._setup_dock_widgets()
            
            # 状态栏
            self._setup_status_bar()
        
        def _setup_menu_bar(self):
            """设置菜单栏"""
            menubar = self.menuBar()
            
            # 文件菜单
            file_menu = menubar.addMenu("文件")
            
            new_action = QAction("新建", self)
            new_action.setShortcut("Ctrl+N")
            file_menu.addAction(new_action)
            
            open_action = QAction("打开...", self)
            open_action.setShortcut("Ctrl+O")
            file_menu.addAction(open_action)
            
            save_action = QAction("保存", self)
            save_action.setShortcut("Ctrl+S")
            file_menu.addAction(save_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction("退出", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # 插件菜单
            plugin_menu = menubar.addMenu("插件")
            
            load_action = QAction("加载插件...", self)
            load_action.triggered.connect(self.load_plugin)
            plugin_menu.addAction(load_action)
            
            unload_action = QAction("卸载插件", self)
            unload_action.triggered.connect(self.unload_plugin)
            plugin_menu.addAction(unload_action)
            
            plugin_menu.addSeparator()
            
            refresh_action = QAction("刷新插件列表", self)
            refresh_action.triggered.connect(self.refresh_plugins)
            plugin_menu.addAction(refresh_action)
            
            # 视图菜单
            view_menu = menubar.addMenu("视图")
            
            self.memory_dock_action = QAction("内存监控", self)
            self.memory_dock_action.setCheckable(True)
            self.memory_dock_action.setChecked(True)
            self.memory_dock_action.triggered.connect(self.toggle_memory_dock)
            view_menu.addAction(self.memory_dock_action)
            
            self.console_dock_action = QAction("控制台", self)
            self.console_dock_action.setCheckable(True)
            self.console_dock_action.setChecked(True)
            self.console_dock_action.triggered.connect(self.toggle_console_dock)
            view_menu.addAction(self.console_dock_action)
            
            self.benchmark_dock_action = QAction("性能测试", self)
            self.benchmark_dock_action.setCheckable(True)
            self.benchmark_dock_action.setChecked(True)
            self.benchmark_dock_action.triggered.connect(self.toggle_benchmark_dock)
            view_menu.addAction(self.benchmark_dock_action)
            
            # 帮助菜单
            help_menu = menubar.addMenu("帮助")
            
            about_action = QAction("关于", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
        
        def _setup_tool_bar(self):
            """设置工具栏"""
            toolbar = QToolBar("主工具栏")
            toolbar.setIconSize(QSize(24, 24))
            self.addToolBar(Qt.TopToolBarArea, toolbar)
            
            # 插件选择
            self.plugin_combo = QComboBox()
            self.plugin_combo.addItems(self.plugin_manager.list_plugins().keys())
            self.plugin_combo.currentTextChanged.connect(self.on_plugin_changed)
            toolbar.addWidget(QLabel("插件:"))
            toolbar.addWidget(self.plugin_combo)
            
            # 设备选择
            self.device_combo = QComboBox()
            self._init_device_combo()
            toolbar.addWidget(QLabel("设备:"))
            toolbar.addWidget(self.device_combo)
            
            toolbar.addSeparator()
            
            # 精度选择
            self.precision_combo = QComboBox()
            self.precision_combo.addItems([mode.name for mode in PrecisionMode])
            self.precision_combo.currentTextChanged.connect(self.on_precision_changed)
            toolbar.addWidget(QLabel("精度:"))
            toolbar.addWidget(self.precision_combo)
            
            toolbar.addSeparator()
            
            # 插件控制按钮
            self.run_btn = QPushButton(QIcon(self._resource_path("run.png")), "运行")
            self.run_btn.clicked.connect(self.run_plugin)
            toolbar.addWidget(self.run_btn)
            
            self.stop_btn = QPushButton(QIcon(self._resource_path("stop.png")), "停止")
            self.stop_btn.clicked.connect(self.stop_plugin)
            self.stop_btn.setEnabled(False)
            toolbar.addWidget(self.stop_btn)
            
            self.reset_btn = QPushButton(QIcon(self._resource_path("reset.png")), "重置")
            self.reset_btn.clicked.connect(self.reset_plugin)
            toolbar.addWidget(self.reset_btn)
        
        def _setup_plugin_tabs(self):
            """设置插件标签页"""
            # 主页标签
            home_tab = QWidget()
            home_layout = QVBoxLayout()
            home_tab.setLayout(home_layout)
            
            # 欢迎信息
            welcome_label = QLabel("<h1>GPU加速插件系统</h1>")
            welcome_label.setAlignment(Qt.AlignCenter)
            home_layout.addWidget(welcome_label)
            
            # 插件信息
            self.plugin_info = QTextEdit()
            self.plugin_info.setReadOnly(True)
            home_layout.addWidget(self.plugin_info)
            
            self.tab_widget.addTab(home_tab, "主页")
            
            # 添加其他插件标签页
            # 在实际应用中，这里会根据插件动态添加标签页
        
        def _setup_dock_widgets(self):
            """设置停靠窗口"""
            # 内存监控
            self.memory_dock = MemoryMonitorDock(self)
            self.addDockWidget(Qt.RightDockWidgetArea, self.memory_dock)
            
            # 控制台
            self.console_dock = PluginConsoleDock(self)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
            
            # 性能测试
            self.benchmark_dock = BenchmarkResultsDock(self)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.benchmark_dock)
        
        def _setup_status_bar(self):
            """设置状态栏"""
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            
            # 插件状态
            self.plugin_status = QLabel("就绪")
            self.status_bar.addPermanentWidget(self.plugin_status)
            
            # 进度条
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximumWidth(200)
            self.progress_bar.setRange(0, 100)
            self.status_bar.addPermanentWidget(self.progress_bar)
            self.progress_bar.hide()
        
        def _init_device_combo(self):
            """初始化设备选择框"""
            self.device_combo.clear()
            devices = self.plugin_manager.list_devices()
            
            for dev_id, info in devices.items():
                text = f"{dev_id}: {info['name']} ({info['total_memory']/1024**3:.1f}GB)"
                self.device_combo.addItem(text, dev_id)
        
        def on_plugin_changed(self, plugin_name: str):
            """插件选择变化"""
            try:
                if self.current_plugin:
                    self.current_plugin.cleanup()
                    
                self.current_plugin = self.plugin_manager.create_instance(
                    plugin_name,
                    self.device_combo.currentData()
                )
                
                # 更新UI
                self.update_plugin_info()
                self.plugin_status.setText(f"插件 '{plugin_name}' 已加载")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载插件: {str(e)}")
                self.plugin_status.setText("插件加载失败")
        
        def on_precision_changed(self, precision: str):
            """精度模式变化"""
            if not self.current_plugin:
                return
                
            try:
                self.current_plugin.precision = PrecisionMode[precision]
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法更改精度模式: {str(e)}")
        
        def update_plugin_info(self):
            """更新插件信息显示"""
            if not self.current_plugin:
                self.plugin_info.clear()
                return
                
            info = self.plugin_manager.get_plugin_info(self.plugin_combo.currentText())
            
            info_text = (
                f"<h2>{info['name']}</h2>"
                f"<p><b>类名:</b> {info['class']}</p>"
                f"<p><b>描述:</b> {info.get('doc', '无描述')}</p>"
                f"<p><b>依赖:</b> {', '.join(info['dependencies']) or '无'}</p>"
                "<h3>可用方法:</h3>"
            )
            
            for method, details in info['methods'].items():
                info_text += (
                    f"<p><b>{method}{details['signature']}</b><br>"
                    f"{details.get('doc', '无文档')}</p>"
                )
            
            self.plugin_info.setHtml(info_text)
        
        def run_plugin(self):
            """运行当前插件"""
            if not self.current_plugin:
                QMessageBox.warning(self, "警告", "没有活动的插件")
                return
                
            # 在实际应用中，这里会调用插件的特定方法
            # 这里只是一个示例
            try:
                self.run_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.progress_bar.show()
                
                worker = PluginWorker(self.current_plugin, "benchmark")
                worker.signals.finished.connect(self.on_worker_finished)
                worker.signals.error.connect(self.on_worker_error)
                worker.signals.progress.connect(self.progress_bar.setValue)
                worker.signals.memory_usage.connect(self.memory_dock.update_memory_usage)
                
                self.workers.append(worker)
                worker.start()
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法启动工作线程: {str(e)}")
                self.reset_ui_state()
        
        def stop_plugin(self):
            """停止当前插件操作"""
            if self.workers:
                for worker in self.workers:
                    worker.stop()
        
        def reset_plugin(self):
            """重置当前插件"""
            if self.current_plugin:
                try:
                    self.current_plugin.reset()
                    QMessageBox.information(self, "成功", "插件已重置")
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"无法重置插件: {str(e)}")
        
        def on_worker_finished(self, result):
            """工作线程完成"""
            self.reset_ui_state()
            
            if isinstance(result, BenchmarkResult):
                self.benchmark_dock.update_results([result])
            
            QMessageBox.information(self, "完成", "操作成功完成")
        
        def on_worker_error(self, error_msg):
            """工作线程错误"""
            self.reset_ui_state()
            QMessageBox.critical(self, "错误", error_msg)
        
        def reset_ui_state(self):
            """重置UI状态"""
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.hide()
            self.workers.clear()
        
        def load_plugin(self):
            """加载插件"""
            # 在实际应用中，这里会从文件加载插件
            QMessageBox.information(self, "信息", "插件加载功能将在完整版中实现")
        
        def unload_plugin(self):
            """卸载插件"""
            if not self.current_plugin:
                QMessageBox.warning(self, "警告", "没有活动的插件")
                return
                
            plugin_name = self.plugin_combo.currentText()
            self.plugin_manager.unregister_plugin(plugin_name)
            self.plugin_combo.removeItem(self.plugin_combo.currentIndex())
            self.current_plugin = None
            
            QMessageBox.information(self, "成功", f"插件 '{plugin_name}' 已卸载")
        
        def refresh_plugins(self):
            """刷新插件列表"""
            self.plugin_combo.clear()
            self.plugin_combo.addItems(self.plugin_manager.list_plugins().keys())
            QMessageBox.information(self, "成功", "插件列表已刷新")
        
        def toggle_memory_dock(self, visible: bool):
            """切换内存监控停靠窗口"""
            self.memory_dock.setVisible(visible)
            self.memory_dock_action.setChecked(visible)
        
        def toggle_console_dock(self, visible: bool):
            """切换控制台停靠窗口"""
            self.console_dock.setVisible(visible)
            self.console_dock_action.setChecked(visible)
        
        def toggle_benchmark_dock(self, visible: bool):
            """切换性能测试停靠窗口"""
            self.benchmark_dock.setVisible(visible)
            self.benchmark_dock_action.setChecked(visible)
        
        def show_about(self):
            """显示关于对话框"""
            about_text = (
                "<h1>GPU加速插件系统</h1>"
                "<p>版本: 1.0.0</p>"
                "<p>基于Python和CUDA的GPU加速插件框架</p>"
                "<p>© 2023 版权所有</p>"
            )
            QMessageBox.about(self, "关于", about_text)
        
        def closeEvent(self, event):
            """关闭事件处理"""
            # 停止所有工作线程
            for worker in self.workers:
                worker.stop()
                worker.wait(1000)
            
            # 清理插件
            if self.current_plugin:
                self.current_plugin.shutdown()
            
            # 保存设置
            self._save_settings()
            
            event.accept()

    def run_gui():
        """运行GUI应用程序"""
        # 设置高DPI支持
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        app = QApplication(sys.argv)
        
        # 设置应用程序样式
        app.setStyle('Fusion')
        
        # 创建并显示主窗口
        window = PluginMainWindow()
        window.show()
        
        # 运行应用程序
        sys.exit(app.exec_())

# ==================== 示例插件 ====================

class MatrixOperations(GPUPluginBase):
    """GPU加速的矩阵运算插件"""
    
    def __init__(self, device_id: int = 0, config: Optional[Dict[str, Any]] = None):
        super().__init__(device_id, config)
        self._stream = cp.cuda.Stream()
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        矩阵乘法
        
        参数:
            a: 第一个矩阵 (m x n)
            b: 第二个矩阵 (n x p)
            
        返回:
            乘积矩阵 (m x p)
        """
        if a.shape[1] != b.shape[0]:
            raise GPUPluginError("矩阵维度不匹配")
        
        # 在CPU回退模式下跳过内存检查
        if not isinstance(cp, DummyCuPy):
            # 检查内存需求
            required_mem = a.nbytes + b.nbytes + a.shape[0] * b.shape[1] * a.itemsize
            if not self.check_memory(required_mem):
                raise MemoryAllocationError("GPU内存不足")
        
        # 在CPU回退模式下直接使用numpy
        if isinstance(cp, DummyCuPy):
            return np.matmul(a, b)
        
        # 传输数据到GPU
        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)
        
        # 执行计算
        with self._stream:
            result = a_gpu @ b_gpu
        
        self._stream.synchronize()
        
        # 返回结果到CPU
        return self.to_cpu(result)

    
    def batch_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        批量矩阵乘法
        
        参数:
            a: 第一个矩阵批次 (batch x m x n)
            b: 第二个矩阵批次 (batch x n x p)
            
        返回:
            乘积矩阵批次 (batch x m x p)
        """
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
            raise GPUPluginError("矩阵批次或维度不匹配")
            
        # 传输数据到GPU
        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)
        
        # 使用einsum进行批量矩阵乘法
        with self._stream:
            result = cp.einsum('bij,bjk->bik', a_gpu, b_gpu)
        
        self._stream.synchronize()
        
        return self.to_cpu(result)
    
    def svd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        奇异值分解
        
        参数:
            matrix: 输入矩阵 (m x n)
            
        返回:
            U, S, Vh: 分解结果
        """
        matrix_gpu = self.to_gpu(matrix)
        
        with self._stream:
            U, S, Vh = cp.linalg.svd(matrix_gpu, full_matrices=False)
        
        self._stream.synchronize()
        
        return self.to_cpu(U), self.to_cpu(S), self.to_cpu(Vh)

# ==================== 主程序入口 ====================

def main():
    """主程序入口"""
    if len(sys.argv) > 1 and sys.argv[1] == "--gui" and QT_AVAILABLE:
        run_gui()
    else:
        # 命令行模式
        print("GPU加速插件系统 - 命令行模式")
        print("使用 --gui 参数启动图形界面")
        
        # 示例使用
        manager = PluginManager()
        manager.register_plugin("matrix_ops", MatrixOperations)
        
        plugin = manager.create_instance("matrix_ops")
        
        # 示例矩阵乘法
        a = np.random.rand(100, 100).astype(np.float32)
        b = np.random.rand(100, 100).astype(np.float32)
        
        result = plugin.matmul(a, b)
        print("矩阵乘法结果形状:", result.shape)
        
        plugin.shutdown()

if __name__ == "__main__":
    main()