from abc import ABC, abstractmethod
from concurrent.futures import thread
from enum import Enum, auto
from pathlib import Path
import threading
from typing import Optional, Type

from Module.io.Camera import Camera

class ProtocolState(Enum):
    IDLE = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()

class Protocol(ABC):
    '''main class for defining a protocol to be run'''
    def __init__(self, camera: Camera) -> None:
        self._state = ProtocolState.IDLE
        self._thread: Optional[threading.Thread] = None
        self.camera: Camera = camera
    
    @property
    def state(self) -> ProtocolState:
        return self._state
    
    def start(self) -> None:
        if self._state != ProtocolState.IDLE:
            raise RuntimeError("Protocol is currently running")
        
        self._state = ProtocolState.RUNNING
        self._thread = threading.Thread(target = self._start_protocol, daemon=True)
        self._thread.start()
        
        
    def stop(self) -> None:
        if self._state == ProtocolState.RUNNING:
            self._state = ProtocolState.STOPPING
    
    def _start_protocol(self):
        try:
            self._run()
            self._state = ProtocolState.IDLE
        except Exception:
            self._state = ProtocolState.ERROR
            raise
        finally:
            if self._thread:
                self._thread.join()
    
    @abstractmethod            
    def _run(self):
        pass