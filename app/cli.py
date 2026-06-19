#!/usr/bin/env python3

import click
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import sqlite3

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from app.config import settings
from app.database import SessionLocal, VideoMetadata, init_db
from app.utils import get_video_properties, format_size, compute_file_hash
from app.optimization import MemoryOptimizer, GPUOptimizer, DynamicConfig
from app.analytics import ReportGenerator

@click.group()
def cli():
    """Video Analytics Management Tool"""
    pass

# ====================== VIDEO COMMANDS ======================

@cli.group()
def video():
    """Manage videos"""
    pass

@video.command()
def list():
    """List all videos"""
    db = SessionLocal()
    videos = db.query(VideoMetadata).all()
    db.close()
    
    if not videos:
        click.echo("No videos found")
        return
    
    click.echo(f"\n{'ID':<20} {'Hash':<10} {'Uploaded':<20}")
    click.echo("-" * 50)
    
    for video in videos:
        click.echo(f"{video.filename:<20} {video.file_hash[:8]:<10} {video.upload_time}")

@video.command()
@click.argument('video_id')
def info(video_id):
    """Get video information"""
    db = SessionLocal()
    video = db.query(VideoMetadata).filter(VideoMetadata.filename == video_id).first()
    db.close()
    
    if not video:
        click.echo(f"Video {video_id} not found", err=True)
        return
    
    video_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")
    
    if os.path.exists(video_path):
        props = get_video_properties(video_path)
        size = os.path.getsize(video_path)
        
        click.echo(f"""
Video ID:       {video.filename}
Hash:           {video.file_hash}
Uploaded:       {video.upload_time}
File Size:      {format_size(size)}
Resolution:     {props.get('width')}x{props.get('height')}
FPS:            {props.get('fps')}
Frames:         {int(props.get('frame_count'))}
Duration:       {props.get('duration_seconds'):.2f}s
        """)
    else:
        click.echo(f"Video file not found for {video_id}", err=True)

@video.command()
@click.argument('video_id')
@click.confirmation_option(prompt='Are you sure you want to delete this video?')
def delete(video_id):
    """Delete video"""
    db = SessionLocal()
    video = db.query(VideoMetadata).filter(VideoMetadata.filename == video_id).first()
    
    if not video:
        click.echo(f"Video {video_id} not found", err=True)
        db.close()
        return
    
    # Delete files
    for path in [settings.video_storage_path, settings.frame_storage_path]:
        for f in os.listdir(path):
            if video_id in f:
                try:
                    os.remove(os.path.join(path, f))
                except Exception as e:
                    click.echo(f"Error deleting {f}: {e}", err=True)
    
    # Delete from database
    db.delete(video)
    db.commit()
    db.close()
    
    click.echo(f"✅ Deleted {video_id}")

@video.command()
def cleanup():
    """Clean up orphaned files"""
    db = SessionLocal()
    videos = db.query(VideoMetadata).all()
    video_ids = set(v.filename for v in videos)
    db.close()
    
    deleted_count = 0
    
    # Check frames directory
    for f in os.listdir(settings.frame_storage_path):
        video_id = f.split('_')[0] + '_' + f.split('_')[1]
        if video_id not in video_ids:
            try:
                os.remove(os.path.join(settings.frame_storage_path, f))
                deleted_count += 1
            except Exception as e:
                click.echo(f"Error deleting {f}: {e}", err=True)
    
    click.echo(f"✅ Deleted {deleted_count} orphaned files")

# ====================== SYSTEM COMMANDS ======================

@cli.group()
def system():
    """System information and control"""
    pass

@system.command()
def status():
    """System status"""
    memory = MemoryOptimizer.get_memory_info()
    gpu_available = GPUOptimizer.is_gpu_available()
    
    click.echo(f"""
System Status:
  Memory:         {memory['used_gb']:.1f}GB / {memory['total_gb']:.1f}GB ({memory['percent']}%)
  GPU Available:  {gpu_available}
    """)
    
    if gpu_available:
        gpu_mem = MemoryOptimizer.get_gpu_memory_info()
        click.echo(f"  GPU Memory:     {gpu_mem['allocated_gb']:.1f}GB / {gpu_mem['total_gb']:.1f}GB")
        click.echo(f"  GPU Device:     {GPUOptimizer.get_gpu_name()}")
    
    # Database stats
    db = SessionLocal()
    video_count = db.query(VideoMetadata).count()
    db.close()
    
    click.echo(f"""
Database:
  Videos:         {video_count}
  Storage:        {settings.video_storage_path}
    """)

@system.command()
def optimize():
    """Auto-optimize configuration"""
    config = DynamicConfig.auto_configure()
    
    click.echo("Recommended Configuration:")
    for key, value in config.items():
        click.echo(f"  {key}: {value}")

@system.command()
def memory_info():
    """Detailed memory information"""
    mem = MemoryOptimizer.get_memory_info()
    
    click.echo(f"""
Memory Information:
  Total:          {mem['total_gb']:.2f} GB
  Available:      {mem['available_gb']:.2f} GB
  Used:           {mem['used_gb']:.2f} GB
  Usage:          {mem['percent']:.1f}%
    """)
    
    if GPUOptimizer.is_gpu_available():
        gpu_mem = MemoryOptimizer.get_gpu_memory_info()
        click.echo(f"""
GPU Memory:
  Total:          {gpu_mem['total_gb']:.2f} GB
  Allocated:      {gpu_mem['allocated_gb']:.2f} GB
  Reserved:       {gpu_mem['reserved_gb']:.2f} GB
        """)

@system.command()
def cleanup_memory():
    """Clean up memory"""
    MemoryOptimizer.cleanup_memory()
    click.echo("✅ Memory cleanup completed")

# ====================== DATABASE COMMANDS ======================

@cli.group()
def db():
    """Database operations"""
    pass

@db.command()
def init():
    """Initialize database"""
    init_db()
    click.echo("✅ Database initialized")

@db.command()
def stats():
    """Database statistics"""
    db_session = SessionLocal()
    
    video_count = db_session.query(VideoMetadata).count()
    
    total_size = 0
    for video in db_session.query(VideoMetadata).all():
        video_path = os.path.join(settings.video_storage_path, f"{video.filename}.mp4")
        if os.path.exists(video_path):
            total_size += os.path.getsize(video_path)
    
    db_session.close()
    
    click.echo(f"""
Database Statistics:
  Total Videos:   {video_count}
  Total Size:     {format_size(total_size)}
  Avg Size:       {format_size(total_size / video_count) if video_count > 0 else 'N/A'}
    """)

@db.command()
def backup():
    """Backup database"""
    import shutil
    from datetime import datetime
    
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"metadata_{timestamp}.db"
    
    try:
        shutil.copy(settings.database_url.replace("sqlite:///", ""), backup_file)
        click.echo(f"✅ Database backed up to {backup_file}")
    except Exception as e:
        click.echo(f"❌ Backup failed: {e}", err=True)

# ====================== SERVER COMMANDS ======================

@cli.group()
def server():
    """Server management"""
    pass

@server.command()
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def start(host, port, reload):
    """Start API server"""
    click.echo(f"Starting server at {host}:{port}...")
    
    cmd = [
        'python', '-m', 'uvicorn',
        'app.main:app',
        f'--host={host}',
        f'--port={port}'
    ]
    
    if reload:
        cmd.append('--reload')
    
    subprocess.run(cmd)

@server.command()
@click.option('--port', default=8501, help='Port number')
def ui(port):
    """Start UI"""
    click.echo(f"Starting UI at localhost:{port}...")
    subprocess.run(['streamlit', 'run', 'app/ui.py', f'--server.port={port}'])

@server.command()
def all():
    """Start all services"""
    click.echo("Starting all services...")
    
    # Start backend
    backend_proc = subprocess.Popen([
        'python', '-m', 'uvicorn',
        'app.main:app',
        '--host=0.0.0.0',
        '--port=8000'
    ])
    
    import time
    time.sleep(3)
    
    # Start frontend
    subprocess.Popen(['streamlit', 'run', 'app/ui.py', '--server.port=8501'])
    
    click.echo("✅ All services started")
    click.echo("API:  http://localhost:8000")
    click.echo("UI:   http://localhost:8501")
    
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        backend_proc.terminate()
        click.echo("\n🛑 Services stopped")

# ====================== TESTING COMMANDS ======================

@cli.group()
def test():
    """Run tests"""
    pass

@test.command()
def pipeline():
    """Run test pipeline"""
    click.echo("Running test pipeline...")
    subprocess.run(['python', '-m', 'app.test_pipeline'])

@test.command()
def health():
    """Health check"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            click.echo(f"✅ API Health: {data.get('overall', 'unknown')}")
        else:
            click.echo(f"❌ API returned status {response.status_code}", err=True)
    except Exception as e:
        click.echo(f"❌ Cannot connect to API: {e}", err=True)

# ====================== MAIN ======================

if __name__ == '__main__':
    cli()
