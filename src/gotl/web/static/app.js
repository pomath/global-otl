/* gotl dashboard — minimal JS (HTMX handles most interactivity) */

function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => {
        el.style.display = 'none';
    });
    document.getElementById(tabId).style.display = 'block';

    // Trigger HTMX to load lazy content
    var target = document.getElementById(tabId);
    htmx.trigger(target, 'revealed');
}

function toggleJobFields() {
    var cmd = document.getElementById('job-command').value;
    var otl = document.getElementById('fields-otl');
    var rinex = document.getElementById('fields-rinex');
    var hardisp = document.getElementById('fields-hardisp');
    if (otl) otl.style.display = (cmd === 'gen-otl-params') ? '' : 'none';
    if (rinex) rinex.style.display = (cmd === 'process-rinex' || cmd === 'download-rinex') ? '' : 'none';
    if (hardisp) hardisp.style.display = (cmd === 'gen-hardisp') ? '' : 'none';
}

function loadLog(jobId) {
    var viewer = document.getElementById('log-viewer');
    if (viewer) {
        viewer.innerHTML = '<p aria-busy="true">Loading log...</p>';
        viewer.setAttribute('hx-get', '/api/jobs/' + jobId + '/log');
        viewer.setAttribute('hx-trigger', 'load, every 5s');
        htmx.process(viewer);
        htmx.trigger(viewer, 'load');
    }
}

/* Preserve log viewer scroll position across HTMX polls.
   Auto-scrolls to bottom if user was already at/near the bottom,
   otherwise restores the previous scroll position. */
(function() {
    var savedScroll = null;
    var wasAtBottom = true;

    document.addEventListener('htmx:beforeSwap', function(evt) {
        var viewer = document.getElementById('log-viewer');
        if (!viewer || !viewer.contains(evt.detail.target)) return;
        var pre = viewer.querySelector('pre');
        if (pre) {
            savedScroll = pre.scrollTop;
            wasAtBottom = (pre.scrollHeight - pre.scrollTop - pre.clientHeight) < 30;
        }
    });

    document.addEventListener('htmx:afterSwap', function(evt) {
        var viewer = document.getElementById('log-viewer');
        if (!viewer || !viewer.contains(evt.detail.target)) return;
        var pre = viewer.querySelector('pre');
        if (pre) {
            if (wasAtBottom) {
                pre.scrollTop = pre.scrollHeight;
            } else if (savedScroll !== null) {
                pre.scrollTop = savedScroll;
            }
        }
    });
})();
