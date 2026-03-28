const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/g);
if (scriptMatch) {
    scriptMatch.forEach((scriptTag, idx) => {
        const js = scriptTag.replace(/<script>|<\/script>/g, '');
        try {
            // Using Function constructor to parse the text as JS
            new Function(js);
            console.log(`Script ${idx} syntax OK`);
        } catch (e) {
            console.log(`Script ${idx} Error:`, e.name, e.message);
        }
    });
} else {
    console.log("No scripts found");
}
