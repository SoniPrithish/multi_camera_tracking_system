/**
 * Multi-Camera Tracking Dashboard JavaScript
 */

class Dashboard {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.stats = {};
        this.events = [];
        this.cameras = {};
        this.startTime = Date.now();
        
        this.init();
    }
    
    init() {
        this.connectWebSocket();
        this.startUptimeTimer();
        this.fetchInitialData();
        
        // Periodic refresh
        setInterval(() => this.refreshStats(), 5000);
    }
    
    // WebSocket Connection
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('connected');
                this.ws.send(JSON.stringify({ type: 'subscribe' }));
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateConnectionStatus('disconnected');
            setTimeout(() => this.connectWebSocket(), this.reconnectInterval);
        }
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'track_update':
                this.handleTrackUpdate(message);
                break;
            case 'event':
                this.handleEvent(message.event);
                break;
            case 'stats_update':
                this.handleStatsUpdate(message.data);
                break;
            case 'pong':
                // Heartbeat response
                break;
        }
    }
    
    handleTrackUpdate(data) {
        const { camera_id, tracks } = data;
        this.cameras[camera_id] = tracks;
        this.updateCameraCard(camera_id, tracks);
    }
    
    handleEvent(event) {
        this.events.unshift(event);
        if (this.events.length > 100) {
            this.events.pop();
        }
        this.updateEventTimeline();
    }
    
    handleStatsUpdate(stats) {
        this.stats = stats;
        this.updateStats();
    }
    
    // UI Updates
    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connection-status');
        const dot = statusEl.querySelector('.status-dot');
        const text = statusEl.querySelector('.status-text');
        
        dot.className = 'status-dot ' + status;
        text.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    updateStats() {
        const globalStats = this.stats.global || {};
        
        document.getElementById('total-people').textContent = 
            globalStats.total_unique_people || 0;
        document.getElementById('current-count').textContent = 
            globalStats.current_total_occupancy || 0;
        document.getElementById('zone-entries').textContent = 
            globalStats.total_zone_entries || 0;
        document.getElementById('line-crossings').textContent = 
            globalStats.total_line_crossings || 0;
        document.getElementById('events-count').textContent = 
            `${this.stats.total_events || 0} events`;
        
        this.updateZoneStats();
    }
    
    updateZoneStats() {
        const container = document.getElementById('zone-stats');
        const zones = this.stats.zones || {};
        
        if (Object.keys(zones).length === 0) {
            container.innerHTML = '<p class="empty-state">No zones configured</p>';
            return;
        }
        
        container.innerHTML = Object.entries(zones).map(([id, zone]) => `
            <div class="zone-item">
                <span class="zone-name">${zone.zone_name || id}</span>
                <span class="zone-occupancy">${zone.current_occupancy || 0} people</span>
            </div>
        `).join('');
    }
    
    updateEventTimeline() {
        const container = document.getElementById('event-timeline');
        
        if (this.events.length === 0) {
            container.innerHTML = '<p class="empty-state">No events yet</p>';
            return;
        }
        
        container.innerHTML = this.events.slice(0, 20).map(event => {
            const time = new Date(event.timestamp * 1000).toLocaleTimeString();
            const type = event.event_type;
            let cssClass = '';
            let text = '';
            
            switch (type) {
                case 'zone_entry':
                    cssClass = 'entry';
                    text = `Person entered ${event.metadata?.zone_name || 'zone'}`;
                    break;
                case 'zone_exit':
                    cssClass = 'exit';
                    text = `Person exited ${event.metadata?.zone_name || 'zone'}`;
                    break;
                case 'line_cross':
                    cssClass = 'cross';
                    text = `Crossed ${event.metadata?.line_name || 'line'}`;
                    break;
                case 'camera_entry':
                    cssClass = 'entry';
                    text = `Appeared in ${event.camera_id}`;
                    break;
                case 'camera_exit':
                    cssClass = 'exit';
                    text = `Left ${event.camera_id}`;
                    break;
                default:
                    text = type;
            }
            
            return `
                <div class="event-item ${cssClass}">
                    <span class="event-time">${time}</span>
                    <span class="event-text">${text}</span>
                </div>
            `;
        }).join('');
    }
    
    updateCameraCard(cameraId, tracks) {
        // This would update individual camera cards if we had video feeds
        const grid = document.getElementById('camera-grid');
        let card = document.querySelector(`[data-camera="${cameraId}"]`);
        
        if (!card) {
            // Create new camera card
            card = document.createElement('div');
            card.className = 'camera-card';
            card.setAttribute('data-camera', cameraId);
            card.innerHTML = `
                <div class="camera-header">
                    <span class="camera-name">${cameraId}</span>
                    <span class="camera-status">Live</span>
                </div>
                <div class="camera-feed">
                    <span class="track-count">${tracks.length} tracks</span>
                </div>
                <div class="camera-stats">
                    <span>Tracks: ${tracks.length}</span>
                </div>
            `;
            
            // Remove placeholder if exists
            const placeholder = grid.querySelector('.camera-placeholder');
            if (placeholder) {
                placeholder.remove();
            }
            
            grid.appendChild(card);
        } else {
            card.querySelector('.track-count').textContent = `${tracks.length} tracks`;
            card.querySelector('.camera-stats').innerHTML = `<span>Tracks: ${tracks.length}</span>`;
        }
        
        this.updateCameraCount();
    }
    
    updateCameraCount() {
        const count = document.querySelectorAll('.camera-card').length;
        document.getElementById('camera-count').textContent = `${count} camera${count !== 1 ? 's' : ''}`;
    }
    
    startUptimeTimer() {
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const hours = Math.floor(elapsed / 3600).toString().padStart(2, '0');
            const minutes = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('uptime').textContent = `${hours}:${minutes}:${seconds}`;
        }, 1000);
    }
    
    // API Calls
    async fetchInitialData() {
        try {
            const [streams, stats, events] = await Promise.all([
                this.fetchStreams(),
                this.fetchStats(),
                this.fetchEvents()
            ]);
            
            this.updateFromStreams(streams);
            this.stats = stats;
            this.events = events;
            
            this.updateStats();
            this.updateEventTimeline();
        } catch (error) {
            console.error('Failed to fetch initial data:', error);
        }
    }
    
    async fetchStreams() {
        const response = await fetch('/api/streams');
        const data = await response.json();
        return data.streams || {};
    }
    
    async fetchStats() {
        const response = await fetch('/api/analytics/summary');
        return await response.json();
    }
    
    async fetchEvents() {
        const response = await fetch('/api/events?limit=50');
        const data = await response.json();
        return data.events || [];
    }
    
    async refreshStats() {
        try {
            const stats = await this.fetchStats();
            this.stats = stats;
            this.updateStats();
        } catch (error) {
            console.error('Failed to refresh stats:', error);
        }
    }
    
    updateFromStreams(streams) {
        const grid = document.getElementById('camera-grid');
        
        if (Object.keys(streams).length === 0) {
            return;
        }
        
        // Remove placeholder
        const placeholder = grid.querySelector('.camera-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        Object.entries(streams).forEach(([cameraId, health]) => {
            let card = document.querySelector(`[data-camera="${cameraId}"]`);
            
            if (!card) {
                card = document.createElement('div');
                card.className = 'camera-card';
                card.setAttribute('data-camera', cameraId);
                grid.appendChild(card);
            }
            
            const statusClass = health.status === 'connected' ? '' : 'disconnected';
            
            card.innerHTML = `
                <div class="camera-header">
                    <span class="camera-name">${cameraId}</span>
                    <span class="camera-status ${statusClass}">${health.status}</span>
                </div>
                <div class="camera-feed">
                    <span class="track-count">--</span>
                </div>
                <div class="camera-stats">
                    <span>FPS: ${health.fps.toFixed(1)}</span>
                    <span>Frames: ${health.frame_count}</span>
                    <span>Drops: ${health.drop_count}</span>
                </div>
            `;
        });
        
        this.updateCameraCount();
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});

