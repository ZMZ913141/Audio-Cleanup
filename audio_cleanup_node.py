"""
@title: 音频增强处理模块（专业版）
@author: ComfyUI-Index-TTS
@description: 对TTS生成的音频进行清理和增强，支持降噪、滤波、归一化、动态压缩、高频增强、音量调节、可调混响等功能
"""

import os
import sys
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal as scipy_signal
from scipy.signal import wiener

# 确保当前目录在导入路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class AudioCleanupNode:
    """
    ComfyUI的音频清理节点，用于去除杂音或添加混响，提高人声质量
    支持中文界面、参数提示、可调混响效果
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"tooltip": "需要处理的原始音频输入"}),
                "降噪强度": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "控制降噪程度，值越大降噪越强，但可能影响音质"}),
                "去混响强度": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "控制混响消除程度，完全人声建议设为1.0"}),
                "高通滤波频率": ("FLOAT", {"default": 80.0, "min": 20.0, "max": 500.0, "step": 10.0,
                                          "tooltip": "过滤低频噪音，提高语音清晰度"}),
                "低通滤波频率": ("FLOAT", {"default": 9000.0, "min": 1000.0, "max": 16000.0, "step": 100.0,
                                          "tooltip": "过滤高频噪音，减少嘶嘶声"}),
                "归一化": (["true", "false"], {"default": "true",
                                            "tooltip": "将音频幅度标准化到[-1,1]范围"}),
                "压缩动态范围": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1,
                                         "tooltip": "增强较安静部分的声音，适用于录音音量较小的情况。0表示不启用"}),
                "提升高频": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1,
                                     "tooltip": "增强高频成分，使声音更明亮清晰。0表示不启用"}),
                "音量增益_dB": ("FLOAT", {"default": 1.0, "min": -6.0, "max": 12.0, "step": 1.0,
                                        "tooltip": "调整音频整体音量，单位为分贝(dB)。正值增大音量，负值减小音量"}),
                "自动增益控制": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                                         "tooltip": "智能调整音量以达到最佳效果，0表示不启用。值越大，增益越高"}),
                "添加回响": (
                    ["false", "小房间", "大房间", "大厅", "山洞", "浴室", "走廊", "山谷", "教堂"],
                    {
                        "default": "false",
                        "tooltip": "选择空间类型添加不同风格的混响效果"
                    }
                ),
                "混响时间": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1,
                                      "tooltip": "混响的持续时间（秒），数值越大声音越“空旷”"}),
                "混响强度": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "混响信号在最终音频中所占比例"}),
                "衰减速度": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "控制混响回声的衰减速度，数值越高衰减越快"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("增强音频",)
    FUNCTION = "enhance_audio"
    CATEGORY = "audio/音频处理"
    DESCRIPTION = "对输入音频进行降噪、去混响和其他增强处理，提高语音质量"

    def __init__(self):
        print("[音频清理] 初始化音频清理节点")

    def enhance_audio(self, 音频, 降噪强度=0.3, 去混响强度=0.6,
                     高通滤波频率=80.0, 低通滤波频率=9000.0, 归一化="true",
                     压缩动态范围=0.1, 提升高频=0.1, 音量增益_dB=1.0,
                     自动增益控制=0.0, 添加回响="false", 混响时间=0.5,
                     混响强度=0.3, 衰减速度=0.8):
        """
        增强音频质量，去除杂音或添加混响
        
        参数:
            音频: ComfyUI音频格式，字典包含 "waveform" 和 "sample_rate"
            其他参数见描述
        返回:
            enhanced_audio: 增强后的音频，ComfyUI音频格式
        """
        try:
            print(f"[音频清理] 开始处理音频")

            if isinstance(音频, dict) and "waveform" in 音频 and "sample_rate" in 音频:
                waveform = 音频["waveform"]
                sample_rate = 音频["sample_rate"]

                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.squeeze(0).cpu().numpy()

                if waveform.ndim > 1 and waveform.shape[0] > 1:
                    print(f"[音频清理] 检测到多通道音频({waveform.shape[0]}通道)，使用第一个通道")
                    audio_data = waveform[0]
                else:
                    audio_data = waveform.flatten()

                print(f"[音频清理] 处理前的音频形状: {audio_data.shape}")

                enhanced_audio = audio_data.copy()

                # 高通滤波
                if 高通滤波频率 > 20.0:
                    print(f"[音频清理] 应用高通滤波器，截止频率: {高通滤波频率}Hz")
                    b, a = scipy_signal.butter(6, 高通滤波频率 / (sample_rate / 2), 'highpass')
                    enhanced_audio = scipy_signal.filtfilt(b, a, enhanced_audio)

                # 低通滤波
                if 低通滤波频率 < 16000.0:
                    print(f"[音频清理] 应用低通滤波器，截止频率: {低通滤波频率}Hz")
                    b, a = scipy_signal.butter(6, 低通滤波频率 / (sample_rate / 2), 'lowpass')
                    enhanced_audio = scipy_signal.filtfilt(b, a, enhanced_audio)

                # Wiener 滤波辅助去噪
                enhanced_audio = wiener(enhanced_audio)

                # 降噪
                if 降噪强度 > 0.1:
                    print(f"[音频清理] 应用降噪处理，强度: {降噪强度}")
                    stft = librosa.stft(enhanced_audio, n_fft=2048, hop_length=512, win_length=2048)
                    noise_stft = np.abs(stft[:, :int(stft.shape[1] * 0.1)])
                    noise_spec = np.mean(noise_stft, axis=1)
                    spec = np.abs(stft)
                    phase = np.angle(stft)
                    noise_spec_expanded = noise_spec[:, None]
                    spec_sub = np.maximum(spec - np.minimum(降噪强度 * noise_spec_expanded, 0.5 * spec), 0.01 * spec)
                    enhanced_stft = spec_sub * np.exp(1j * phase)
                    enhanced_audio = librosa.istft(enhanced_stft, length=len(enhanced_audio))

                # 动态范围压缩
                if 压缩动态范围 > 0.0:
                    print(f"[音频清理] 应用动态范围压缩，强度: {压缩动态范围}")
                    enhanced_audio = np.sign(enhanced_audio) * (
                        1 - np.exp(-np.abs(enhanced_audio) / (压缩动态范围 + 0.1))
                    )

                # 增强高频
                if 提升高频 > 0.0:
                    print(f"[音频清理] 应用高频增强，强度: {提升高频}")
                    b, a = scipy_signal.butter(4, 6000 / (sample_rate / 2), 'high')
                    high_freq = scipy_signal.filtfilt(b, a, enhanced_audio)
                    enhanced_audio += 提升高频 * high_freq

                # 音量增益
                if 音量增益_dB != 0.0:
                    print(f"[音频清理] 应用音量增益: {音量增益_dB}dB")
                    gain_factor = 10 ** (音量增益_dB / 20)
                    enhanced_audio *= gain_factor

                # AGC
                if 自动增益控制 > 0.0:
                    rms = np.sqrt(np.mean(enhanced_audio ** 2))
                    target_rms = 0.3
                    gain = target_rms / (rms + 1e-6)
                    enhanced_audio *= (自动增益控制 * gain + (1 - 自动增益控制))

                # 添加混响
                if 添加回响 != "false":
                    print(f"[音频清理] 添加混响：{添加回响}，时间={混响时间}s，强度={混响强度}，衰减={衰减速度}")
                    room_impulse = self._generate_room_impulse(
                        sample_rate,
                        reverb_type=添加回响,
                        reverb_time=混响时间,
                        decay_speed=衰减速度
                    )
                    enhanced_audio = scipy_signal.fftconvolve(enhanced_audio, room_impulse, mode='same')

                # 归一化
                if 归一化 == "true":
                    print("[音频清理] 应用音频归一化")
                    enhanced_audio = librosa.util.normalize(enhanced_audio)

                # 转换为torch tensor并返回
                enhanced_tensor = torch.tensor(enhanced_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                enhanced_dict = {
                    "waveform": enhanced_tensor,
                    "sample_rate": sample_rate
                }

                print(f"[音频清理] 音频增强完成，输出形状: {enhanced_tensor.shape}")
                return (enhanced_dict,)
            else:
                raise ValueError("输入音频格式不支持，应为ComfyUI的AUDIO类型")

        except Exception as e:
            import traceback
            print(f"[音频清理] 处理音频失败: {e}")
            traceback.print_exc()

            # 错误时返回警告音
            sample_rate = 24000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            warning_tone = np.sin(2 * np.pi * 880 * t).astype(np.float32)
            signal_tensor = torch.tensor(warning_tone, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audio_dict = {"waveform": signal_tensor, "sample_rate": sample_rate}
            return (audio_dict,)

    def _generate_room_impulse(self, sr, reverb_type="小房间", reverb_time=0.5, decay_speed=0.8):
        """
        根据设定生成不同类型的空间脉冲响应
        """
        t = np.linspace(0, reverb_time, int(sr * reverb_time))

        delays = []
        weights = []

        if reverb_type == "小房间":
            delays = [0, 0.05, 0.1, 0.15]
            weights = [1.0, 0.6, 0.4, 0.2]
        elif reverb_type == "大房间":
            delays = [0, 0.1, 0.25, 0.4, 0.6]
            weights = [1.0, 0.7, 0.5, 0.3, 0.15]
        elif reverb_type == "大厅":
            delays = [0, 0.2, 0.5, 0.8, 1.0, 1.2]
            weights = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
        elif reverb_type == "山洞":
            delays = [0, 0.3, 0.6, 1.0, 1.5]
            weights = [1.0, 0.9, 0.7, 0.5, 0.3]
        elif reverb_type == "浴室":
            delays = [0, 0.03, 0.06, 0.09, 0.12]
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        elif reverb_type == "走廊":
            delays = [0, 0.1, 0.2, 0.3]
            weights = [1.0, 0.7, 0.5, 0.3]
        elif reverb_type == "山谷":
            delays = [0, 0.5, 1.0, 1.5, 2.0]
            weights = [1.0, 0.7, 0.5, 0.3, 0.2]
        elif reverb_type == "教堂":
            delays = [0, 0.4, 0.8, 1.2, 1.6, 2.0]
            weights = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2]
        else:
            return np.zeros(int(sr * 0.1))  # 返回极短零信号表示无混响

        impulse = np.zeros_like(t)
        for delay, weight in zip(delays, weights):
            idx = int(delay * sr)
            if idx < len(impulse):
                impulse[idx] += weight

        decay = np.exp(-t * (decay_speed * 10))
        return np.convolve(impulse, decay, mode='full')[:len(t)]