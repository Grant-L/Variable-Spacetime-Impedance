<?php
require_once("../inc/db.inc");
require_once("../inc/util.inc");
require_once("../inc/news.inc");
require_once("../inc/cache.inc");
require_once("../project/project.inc");

$user = get_logged_in_user(false);
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AVE Alpha Search | Distributed Topological Physics</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0b10;
            --bg-gradient-1: #161824;
            --bg-gradient-2: #0d121b;
            --accent-primary: #00f0ff;
            --accent-secondary: #ff003c;
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --glass-bg: rgba(30, 41, 59, 0.35);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(0, 240, 255, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(255, 0, 60, 0.05), transparent 25%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            line-height: 1.6;
        }

        /* Animated Background Grid */
        .grid-bg {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: 
                linear-gradient(to right, rgba(255,255,255,0.03) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
            transform: perspective(500px) rotateX(60deg) translateY(-100px) translateZ(-200px);
            animation: gridMove 20s linear infinite;
        }

        @keyframes gridMove {
            0% { transform: perspective(500px) rotateX(60deg) translateY(0) translateZ(-200px); }
            100% { transform: perspective(500px) rotateX(60deg) translateY(50px) translateZ(-200px); }
        }

        /* Navbar */
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 4rem;
            background: rgba(10, 11, 16, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--glass-border);
            position: fixed;
            width: 100%;
            top: 0;
            z-index: 100;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #fff, var(--accent-primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-links a {
            color: var(--text-main);
            text-decoration: none;
            margin-left: 2rem;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: color 0.3s ease;
        }

        .nav-links a:hover {
            color: var(--accent-primary);
        }

        /* Main Container */
        .container {
            max-width: 1200px;
            margin: 8rem auto 4rem;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4rem;
            align-items: center;
            flex-grow: 1;
        }

        /* Hero Section */
        .hero-content h1 {
            font-size: 4rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 1.5rem;
        }

        .hero-content h1 span {
            color: var(--accent-primary);
            text-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
        }

        .hero-content p {
            font-size: 1.2rem;
            color: var(--text-muted);
            margin-bottom: 2.5rem;
            max-width: 90%;
        }

        /* Buttons */
        .btn-group {
            display: flex;
            gap: 1.5rem;
        }

        .btn {
            padding: 1rem 2.5rem;
            border-radius: 4px;
            font-size: 1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }

        .btn-primary {
            background: transparent;
            color: var(--accent-primary);
            border: 2px solid var(--accent-primary);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
        }

        .btn-primary:hover {
            background: var(--accent-primary);
            color: #000;
            box-shadow: 0 0 30px rgba(0, 240, 255, 0.6);
            transform: translateY(-2px);
        }

        .btn-secondary {
            background: var(--glass-bg);
            color: var(--text-main);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }

        /* Glass Panel */
        .glass-panel {
            background: var(--glass-bg);
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            padding: 3rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: var(--glass-shadow);
            position: relative;
        }

        .glass-panel::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        }

        .glass-panel h2 {
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #fff;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        .stat-card {
            background: rgba(0,0,0,0.4);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .stat-card h3 {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }

        .stat-card .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            color: var(--accent-primary);
            font-weight: 700;
        }

        /* User Section */
        .user-section {
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        .terminal-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: var(--accent-primary);
            margin-bottom: 1rem;
            display: block;
        }

        .pulse {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: var(--accent-secondary);
            border-radius: 50%;
            margin-right: 8px;
            box-shadow: 0 0 10px var(--accent-secondary);
            animation: blink 1.5s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* Responsive */
        @media (max-width: 968px) {
            .container {
                grid-template-columns: 1fr;
                gap: 3rem;
                margin-top: 6rem;
            }
            .hero-content h1 { font-size: 3rem; }
            nav { padding: 1.5rem 2rem; }
        }
    </style>
</head>
<body>

    <div class="grid-bg"></div>

    <nav>
        <div class="logo">AVE ALPHA</div>
        <div class="nav-links">
            <a href="server_status.php">Status</a>
            <a href="top_users.php">Top Contributors</a>
            <?php if ($user) { ?>
                <a href="home.php">My Account</a>
                <a href="logout.php">Logout</a>
            <?php } else { ?>
                <a href="login_form.php">Login</a>
                <a href="create_account_form.php">Sign Up</a>
            <?php } ?>
        </div>
    </nav>

    <div class="container">
        <div class="hero-content">
            <h1>Derive The<br><span>Fine-Structure</span><br>Constant.</h1>
            <p>Join the distributed computing network searching for the thermodynamic origins of the universe. By analyzing 3D amorphous chiral LC networks using Rigidity Percolation algorithms, we are computationally pinning down the exact value of α (1/137.035999).</p>
            
            <div class="btn-group">
                <?php if (!$user) { ?>
                    <a href="create_account_form.php" class="btn btn-primary">Join The Network</a>
                <?php } else { ?>
                    <a href="download_software.php" class="btn btn-primary">Download Client</a>
                <?php } ?>
                <a href="server_status.php" class="btn btn-secondary">View Metrics</a>
            </div>
        </div>

        <div class="glass-panel">
            <h2><span class="pulse"></span>Master Node Status</h2>
            <span class="terminal-text">> BOINC Daemon: EXECUTING</span>
            <span class="terminal-text">> Matrix Slice Size: 500,000 Nodes</span>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Target Value</h3>
                    <div class="value">137.035...</div>
                </div>
                <div class="stat-card">
                    <h3>Engine</h3>
                    <div class="value">v1.02</div>
                </div>
            </div>

            <div class="user-section">
                <?php if ($user) { ?>
                    <p style="color: #fff; font-size: 1.2rem; margin-bottom: 0.5rem;">Welcome back, <strong><?php echo htmlspecialchars($user->name); ?></strong></p>
                    <p style="color: var(--text-muted); font-size: 0.9rem;">Your contribution helps lock the geometric bounds of physical reality.</p>
                    <div style="margin-top: 1rem;">
                        <span class="terminal-text">> Total Compute Credit: <?php echo format_credit($user->total_credit); ?></span>
                    </div>
                <?php } else { ?>
                    <p style="color: var(--text-muted);">The search requires massive computational topology mapping. A single CPU can process one structural chunk in ~20 minutes. We need millions.</p>
                <?php } ?>
            </div>
        </div>
    </div>

</body>
</html>
